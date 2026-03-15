# pylint: disable=C0114
import time

# pylint: disable=C0114
import mujoco
import mujoco.viewer

# pylint: disable=E1101
m = mujoco.MjModel.from_xml_path("hello.xml")
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
    # close the viewer automatically after 30 wall-seconds.
    start = time.time()
    # while viewer.is_running() and d.time < 10: 
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()
        mujoco.mj_step(m, d)
        viewer.sync()

        # sync to real time
        elapsed = time.time() - step_start
        dt = m.opt.timestep - elapsed
        if dt > 0:
            time.sleep(dt)
