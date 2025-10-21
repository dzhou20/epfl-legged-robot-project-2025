# Mini-Project 1: Quadruped Jumping
Group Member: Xinran Wang, Hsu-li Huang, Dacheng Zhou

---

## 1. Environment Setup

### Create and activate conda environment
```bash
conda create -n quadruped python=3.9
conda activate quadruped
```

### Install dependencies
```bash
pip install -r requirements.txt
```

⚠️ If pybullet fails to install, use conda first:
```bash
conda install conda-forge::pybullet
pip install -r requirements.txt
```

---

## 2. Code Structure

```
lr_mp1_group_21/
├── env/                        # Simulation environment (DO NOT MODIFY)
│   ├── simulation.py           # QuadSimulator class
│   └── utils.py                # Helper functions
│
├── profiles.py                 # Force profile implementation
├── quadruped_jump.py          # Basic jumping controller
├── quadruped_jump_opt.py      # Optimization framework with Optuna
│
├── videos/                     # Generated video recordings
└── requirements.txt           # Python dependencies
```

### Key Files:

- **`profiles.py`**: Defines the `FootForceProfile` class that generates oscillating force patterns for the legs
- **`quadruped_jump.py`**: Implements the basic jumping controller with fixed parameters
- **`quadruped_jump_opt.py`**: Uses Optuna to automatically optimize jumping parameters for different objectives

---

## 3. How to Run

### Basic Controller (Fixed Parameters)

Run the basic jumping controller with predefined parameters:

```bash
python quadruped_jump.py
```

This will:
- Execute a jumping sequence with fixed force parameters
- Display the simulation in PyBullet GUI
- Record a video to the `videos/` directory

#### ⚠️ Important: Configure for Lateral Jumping

By default, `quadruped_jump.py` is configured for **forward jumping** (line 66-73). If you want to test **lateral jumping**, you need to modify the airborne detection logic:

**For Forward Jumping (default):**
```python
# Line 73 in quadruped_jump.py
airborne = 0  # Always grounded mode
```

**For Lateral Jumping:**
Uncomment lines 67-70 and comment out line 73:
```python
# Line 67-73 in quadruped_jump.py
if all(foot_contacts):
    airborne = 0
if not any(foot_contacts):
    airborne = 1

# airborne = 0  # Comment this out for lateral jumping
```

This enables proper airborne detection, which is necessary for lateral jumping stability.

---

### Optimization (Automatic Parameter Tuning)

Run the optimization framework to find the best parameters for different jumping objectives:

```bash
python quadruped_jump_opt.py --objective <objective_name>
```

#### Available Objectives:

##### 1. **Forward Jumping** - Maximize forward distance
```bash
python quadruped_jump_opt.py --objective distance
```
- **Optimizes:** `Fx` (forward force), `Fz` (vertical force), `f0` (frequency)
- **Goal:** Jump as far forward as possible
- **How it works:** All four legs push forward simultaneously

##### 2. **Lateral Jumping** - Maximize sideways distance
```bash
python quadruped_jump_opt.py --objective lateral_distance
```
- **Optimizes:** `Fy` (lateral force), `Fz` (vertical force), `f0` (frequency)
- **Goal:** Jump as far sideways as possible (left or right)
- **How it works:** All four legs push in the same lateral direction

##### 3. **Twist Jumping** - Maximize rotation angle
```bash
python quadruped_jump_opt.py --objective twist
```
- **Optimizes:** `Fy` (lateral force), `Fz` (vertical force), `f0` (frequency)
- **Goal:** Rotate as much as possible while jumping
- **How it works:** Front legs push one direction, rear legs push opposite direction, creating a torque for rotation

##### 4. **Fastest Hopping** - Maximize jumping speed
```bash
python quadruped_jump_opt.py --objective fastest_hopping
```
- **Optimizes:** `Fx` (forward force), `Fz` (vertical force), `f0` (frequency)
- **Goal:** Achieve maximum speed (distance/time) over multiple jumps
- **How it works:** Similar to forward jumping but optimizes for speed rather than distance per jump

##### 5. **Fastest Hopping (Dual Frequency)** - Maximize jumping speed with two frequencies
```bash
python quadruped_jump_opt.py --objective fastest_hopping_dualfreq
```
- **Optimizes:** `Fx` (forward force), `f0` (takeoff frequency), `f1` (landing frequency)
- **Goal:** Achieve maximum speed by optimizing both takeoff and recovery phases
- **How it works:** Unlike standard fastest hopping, this also optimizes the landing/recovery frequency (`f1`)

---

### Optimization Output

After running optimization (50 trials), you will get:

1. **Console output** with best parameters and performance
2. **Visualization**: `optimized_params_vs_trials.png` showing parameter evolution
3. **Video recording**: Best jump automatically replayed and saved to `videos/`

---

### Clean Up Videos

To delete temporary videos (keep only best results):

```bash
cd videos && find . -maxdepth 1 -type f ! -name "best_video*" -delete
```

---

## Key Parameters

- **`Fx`**: Forward force (N) - controls forward motion
- **`Fy`**: Lateral force (N) - controls sideways motion and rotation
- **`Fz`**: Vertical force (N) - controls jump height
- **`f0`**: Takeoff frequency (Hz) - controls force application rate
- **`f1`**: Landing frequency (Hz) - controls recovery between jumps
- **`Katt`**: Virtual model gain (default: 1000) - controls posture stability
