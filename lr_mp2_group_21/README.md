# MP2: Quaduped Locomotion 

This repository contains an environment for simulating locomotion onboard a quadruped robot using cpg alone and also cpg with Reinforcement Learning (RL). The framework is inspired from RL [1] and CPG-RL [2] papers. You will **only modify a few specific files**, the rest provide the environment and helper code.

## Quick start installation 

```bash
# 1. Create environment (with conda, recommended)
conda create -n quadruped python=3.9
conda activate quadruped

# 2. Install dependencies
pip install -r requirements.txt
# If pybullet fails, install through conda before installing requirements again:
conda install conda-forge::pybullet
```

## Not so quick start installation
Installation follows the same structure as the [practicals](https://gitlab.epfl.ch/lgevers/lr-practicals).

### 1. Create a virtual environment
We recommend using **conda** (preferred) or **virtualenv** with **Python 3.9** or higher.
It is a good practice to keep separate environments for different projects, so we encourage you to create a new environment for this project rather than reusing the one for the practicals (especially since the dependencies are not exactly the same).

After downloading the repository, create the virtual environment as follows:

```bash
# With conda (recommended)
conda create -n quadruped python=3.9

# With virtualenv (alternative)
python -m venv venv
```

Then, activate your environment every time you intend on using it for this project:

```bash
# With conda (recommended)
conda activate quadruped

# With virtualenv (alternative)
# Note that you need to be in the project directory containing the venv/ folder
source venv/bin/activate    # Linux/Mac OS
venv\Scripts\Activate       # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

⚠️ If pybullet fails to install (compilation issues), install through conda and then install the `requirements.txt` again:

```bash
conda install conda-forge::pybullet
```

## Repository structure
```bash
.
├── a1_description/             # Model files for a1 robot
│   ├── meshes/                 # Meshes of A1 (DO NOT MODIFY)
│   └── urdf/                   # Robot model structure
├── env/                        # Environment modelling files
│   ├── configs_a1.py           # Configurations of a1 robot components
│   └── hopf_network.py         # CPG class skeleton for various gaits
│   └── quadruped_gym_env.py    # Setting up RL environment 
│   └── quadruped_motor.py      # Motor model of a1 robot
│   └── quadruped.py            # Robot specific functionalities, review this closely to access robot states
│                               # and calling functions to solve inverse kinematics, return the leg Jacobian etc
├── utils/                      # Some file i/o and plotting helpers
│   └── file_utils.py           # Basic helper functions 
│   └── utils.py                # Basic helper functions 
├── load_sb3.py                 # Loads and plays trained RL policy
├── run_cpg.py                  # Run CPG to joint commands
├── run_sb3.py                  # Provides an interface to train RL policy with RL algorithms
└── requirements.txt            # Dependencies
```

## Modify the following files (at the TODO tags)
- `quadruped_gym_env.py`
- `hopf_network.py`
- `run_cpg.py`
- `load_sb3.py`

**These are the main files to modify and get your CPG and CPG-RL pipeline running. It is highly recommended to tune or adjust parameters for better performance from other scripts only when you have setup a basic running CPG and CPG-RL workflow.**

## Code resources
- The [PyBullet Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3) is the current up-to-date documentation for interfacing with the simulation. 
- The quadruped environment took inspiration from [Google's motion-imitation repository](https://github.com/google-research/motion_imitation) based on [this paper](https://xbpeng.github.io/projects/Robotic_Imitation/2020_Robotic_Imitation.pdf). 
- Reinforcement learning algorithms from [stable-baselines3](https://github.com/DLR-RM/stable-baselines3). Also see for example [ray[rllib]](https://github.com/ray-project/ray) and [spinningup](https://github.com/openai/spinningup). You should review the documentation carefully for information on the different algorithms and training hyperparameters. 

## Submission
When submitting zip all your code files together.
These should be your **code files only** (e.g., no `venv` directory for the virtual environment).
You should include the `requirements.txt` in case you are adding new dependencies.
Video namings should be consistent with those that you refer to in your report.
The structure of your submission folder should look like this:

```bash
lr_mp2_group_{group number}.zip
|- report.pdf               # Your report in PDF
|- env/                     # Script with your changes
|  |- configs_a1.py
|  |- hopf_network.py
|  |- quadruped.py
|  |- quadruped_gym_env.py
|  \- quadruped_motor.py
|- utils/                   # Helper scripts
|  |- file_utils.py
|  \- utils.py
|- videos/                  # Include relevant videos!
|  |- TROT_HIGH_0.5ms.mp4|  # Descriptive name examples
|  |- TROT_LOW_0.5ms.mp4|   
|  |- RL_VEL_0.5ms.mp4|
|  |- RL_GAPS.mp4|
|  \- ...                   # Extra videos that you may have
|- weights/                 # Include relevant weights!
|  |- TROT.zip|             # Descriptive name examples
|  |- RL_VEL_0.5ms.zip|
|  |- RL_GAPS.zip|
|  \- ...                   # Extra weights that you may have
|- load_sb3.py              # Script with your changes
|- run_cpg.py               # Script with your changes
|- run_sb3.py               # Script with your changes
\- requirements.txt         # Script with your changes
```

**Note that the max submission size is 100 MB, which means you may have to compress your videos.**

##  References
```
[1] G. Bellegarda and A. Ijspeert, "CPG-RL: Learning Central Pattern Generators for Quadruped Locomotion," in IEEE Robotics and Automation Letters, 2022, doi: 10.1109/LRA.2022.3218167. [IEEE](https://ieeexplore.ieee.org/abstract/document/9932888), [arxiv](https://arxiv.org/abs/2211.00458)

[2] G. Bellegarda, Y. Chen, Z. Liu, and Q. Nguyen, "Robust High-speed Running for Quadruped Robots via Deep Reinforcement Learning," in 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems, 2022. [arxiv](https://arxiv.org/abs/2103.06484)
```

## Tips
- If your simulation is very slow, remove the calls to time.sleep() and disable the camera resets in [quadruped_gym_env.py](./env/quadruped_gym_env.py).
- The camera viewer can be modified in `_render_step_helper()` in [quadruped_gym_env.py](./env/quadruped_gym_env.py) to track the hopper.