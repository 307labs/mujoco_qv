"""Franka Emika Panda IK tracking simulation with MuJoCo viewer."""
import mujoco
import mujoco.viewer
import numpy as np
import time

# ── Trajectory parameters ──
RADIUS = 0.15
CENTER_X = 0.5
CENTER_Y = 0.0
FREQ = 0.5


def circle(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
    """Return (x, y) on a circle of radius r centered at (h, k)."""
    x = r * np.cos(2 * np.pi * f * t) + h
    y = r * np.sin(2 * np.pi * f * t) + k
    return np.array([x, y])


# ── Visual helpers ──
def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Add a capsule segment to the viewer scene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3), np.zeros(3), np.zeros(9),
        rgba.astype(np.float32),
    )
    mujoco.mjv_connector(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius, point1, point2,
    )


def draw_trajectories(scn, target_traj, ee_traj):
    """Draw target (blue) and end-effector (red) traces."""
    blue = np.array([0.0, 0.0, 1.0, 1.0])
    red = np.array([1.0, 0.0, 0.0, 0.8])
    for i in range(len(target_traj) - 1):
        add_visual_capsule(scn, target_traj[i], target_traj[i + 1], 0.005, blue)
        add_visual_capsule(scn, ee_traj[i], ee_traj[i + 1], 0.005, red)


# ── Load model ──
model = mujoco.MjModel.from_xml_path("franka_emika_panda/scene01.xml")
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)

mocap_id = model.body("target").mocapid[0]
site_id = model.site("attachment_site").id

# ── Pre-allocate IK arrays ──
jac = np.zeros((6, model.nv))
error = np.zeros(6)
error_pos = error[:3]
error_ori = error[3:]
site_quat = np.zeros(4)
target_quat_conj = np.zeros(4)
error_quat = np.zeros(4)

# ── Trajectory history (subsample for drawing) ──
target_traj: list[np.ndarray] = []
ee_traj: list[np.ndarray] = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
    viewer.opt.sitegroup[4] = 1

    while viewer.is_running():
        step_start = time.time()

        # ── 1. Move target along circular trajectory ──
        data.mocap_pos[mocap_id, 0:2] = circle(
            data.time, RADIUS, CENTER_X, CENTER_Y, FREQ,
        )

        # ── 2. Compute IK error ──
        # Position error
        error_pos[:] = data.site(site_id).xpos - data.mocap_pos[mocap_id]

        # Orientation error (quaternion → axis-angle)
        mujoco.mju_negQuat(target_quat_conj, data.mocap_quat[mocap_id])
        mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
        mujoco.mju_mulQuat(error_quat, site_quat, target_quat_conj)
        mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)

        # ── 3. Solve differential IK: J·dq = -error ──
        mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
        dq = np.linalg.pinv(jac) @ (-error)

        # ── 4. Apply to position-controlled joints ──
        q = data.qpos.copy()
        mujoco.mj_integratePos(model, q, dq, 1)
        np.clip(q, *model.jnt_range.T, out=q)
        data.ctrl = q

        # ── 5. Step simulation ──
        mujoco.mj_step(model, data)

        # ── 6. Record trajectories (subsample every 10 steps) ──
        target_traj.append(data.mocap_pos[mocap_id].copy())
        ee_traj.append(data.site(site_id).xpos.copy())

        # ── 7. Draw trajectory traces in viewer ──
        with viewer.lock():
            draw_trajectories(
                viewer.user_scn,
                target_traj[::10],
                ee_traj[::10],
            )

        # ── 8. Print status ──
        tgt = data.mocap_pos[mocap_id]
        ee = data.site_xpos[site_id]
        dist = np.linalg.norm(error_pos)
        print(
            f"\rt={data.time:5.2f}s  "
            f"Target:[{tgt[0]:+.3f},{tgt[1]:+.3f},{tgt[2]:+.3f}]  "
            f"EE:[{ee[0]:+.3f},{ee[1]:+.3f},{ee[2]:+.3f}]  "
            f"err:{dist:.4f}",
            end="",
        )

        viewer.sync()

        # ── Real-time sync ──
        elapsed = time.time() - step_start
        dt = model.opt.timestep - elapsed
        if dt > 0:
            time.sleep(dt)
