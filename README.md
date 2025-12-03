# epfl-legged-robot-project-2025

Course mini-projects for the A1 quadruped. The repository holds two separate tasks; each lives in its own folder with its own dependencies.

## MP1: Jump Control (`lr_mp1_group_21`)
- Design foot force profiles and feedback terms to make the robot jump repeatedly in PyBullet (`quadruped_jump.py` is the entry point; `quadruped_jump_opt.py` helps sweep parameters).
- Quick start
  ```bash
  cd lr_mp1_group_21
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  python quadruped_jump.py
  ```
- Assets: `env/` (simulator), `a1_description/` (URDF), `profiles.py` (force profiles), `videos/` (recordings).

## MP2: Locomotion with CPG + RL (`lr_mp2_group_21`)
- Build a Hopf-network CPG gait and optionally refine it with reinforcement learning.
- Main scripts: `run_cpg.py` (hand-tuned CPG), `run_sb3.py` (train RL), `load_sb3.py` (play trained weights). See `lr_mp2_group_21/README.md` for detailed setup.
- Quick start
  ```bash
  cd lr_mp2_group_21
  conda create -n quadruped python=3.9
  conda activate quadruped
  pip install -r requirements.txt   # or conda install conda-forge::pybullet first if needed
  python run_cpg.py                 # or python run_sb3.py
  ```

## Notes
- Each mini-project keeps its own virtual environment; avoid mixing dependencies.
- Rendered simulations are slower; disable GUI to speed up training runs.
