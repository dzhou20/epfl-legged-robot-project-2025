# Learning-based Control for Quadruped Locomotion ⚙️
> Learning-based Control for Quadruped Locomotion 🐕: 
[1] Jumping with Force-Profile Control
[2] Slope/Stair/Gap Locomotion with Hopf Central Pattern Generation (CPG) and Reinforcement Learning (RL)

(Original Course: MICRO-507 Legged Robots)


## Highlights
- MP1: Foot force pulses from a simple CPG oscillator + nominal foot PD + gravity/virtual-model compensation for repeated jumps.
- MP2: Hopf-network CPG generates trot/walk/pace/bound trajectories; optional RL (PPO/SAC via Stable-Baselines3) to refine gait.
- Separate environments and dependencies per task; videos kept out of git; quick-start commands below.

## Repo Structure
- `lr_mp1_group_21/` — Jump control; entry `quadruped_jump.py`; parameter sweep `quadruped_jump_opt.py`; assets `env/`, `a1_description/`, `profiles.py`, `videos/`.
- `lr_mp2_group_21/` — CPG + RL; scripts `run_cpg.py`, `run_sb3.py`, `load_sb3.py`; assets `env/`, `utils/`, `videos/`, `script/`.

## Quick Start
### MP1 (jump)
```bash
cd lr_mp1_group_21
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python quadruped_jump.py          # GUI on
```

### MP2 (CPG / RL)
```bash
cd lr_mp2_group_21
conda create -n quadruped python=3.9
conda activate quadruped
pip install -r requirements.txt
python run_cpg.py                 # pure CPG trot
# train RL (set LEARNING_ALG in run_sb3.py to PPO/SAC as needed)
python run_sb3.py
# play a trained policy
python load_sb3.py --weights path/to/model.zip
```

## Demos
- To be added...

## Algorithms
- MP1: Force-profile oscillator (stance/stage phase) driving foot impulses, mapped via Jacobian to joint torques; nominal foot PD, gravity compensation, virtual-model term when in contact.
- MP2: Hopf CPG in polar form with gait phase offsets -> foot x/z trajectories -> IK + joint/Cartesian PD; RL uses SB3 PPO/SAC to learn/improve CPG-related parameters in PyBullet env.

## Notes
- Keep per-project virtual environments separate to avoid dependency conflicts.
- Rendering slows training; disable GUI/headless where supported for speed.
- Large artifacts (videos, script outputs) are ignored by git; add links or GIFs instead of committing full MP4s.
