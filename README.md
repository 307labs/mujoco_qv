# Overview

## Setup

### Using uv (recommended)
```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/AvQ1301/mujoco_qv.git
cd mujoco_qv
uv sync

# Run
source .venv/bin/activate
python motor_pid.py
```

### Using pip
```bash
python3 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python motor_pid.py
```

# Recommendation
- using wsl2 to code
- ubuntu 22.04
- mujoco 

