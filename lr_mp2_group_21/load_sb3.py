# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

import os, sys
import gymnasium as gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
import datetime
# may be helpful depending on your system
# if platform =="darwin": # mac
#   import PyQt5
#   matplotlib.use("Qt5Agg")
# else: # linux
#   matplotlib.use('TkAgg')

# stable-baselines3
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env # fix for newer versions of stable-baselines3

# utils
from env.quadruped_gym_env import QuadrupedGymEnv
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results

LEARNING_ALG = "PPO" #"SAC"
interm_dir = "./logs/intermediate_models/"
# path to saved models, i.e. interm_dir + 
# log_dir = interm_dir + ''  # velocity tracking task without noise
log_dir = interm_dir + 'medium_space_vel_1.0_exp'  # velocity tracking with noise
video_dir = os.path.join("videos", os.path.basename(log_dir.rstrip("/")))

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
env_config = {
    # "motor_control_mode":"CARTESIAN_PD",  # PD is default, CPG for our project
    "motor_control_mode":"CPG",  # PD is default, CPG for our project
    "task_env": "LR_COURSE_TASK",
    "observation_space_mode": "LR_COURSE_OBS",
    # "observation_space_mode": "FWD_LOCOMOTION",
    "add_noise": False,
    # "terrain": "SLOPES"  # during training, we used slopes terrain
    # add_noise will be set to False to make policy learning faster
}
env_config['render'] = True
env_config['record_video'] = True
env_config['add_noise'] = False
env_config['record_video_dir'] = video_dir

# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
# 如需固定加载某个 checkpoint，可直接指定：
# model_name = os.path.join(log_dir, 'rl_model_2220000_steps')
# for velocity_follow_with_noise, rl_model_1740000_steps (no overfitting) 

monitor_results = load_results(log_dir)
print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
# save the two plots produced by plot_results (figure indices 1 and 2)
plt.figure(1)
plt.savefig(os.path.join(log_dir, 'fig_training_rewards.png'), dpi=200, bbox_inches='tight')
plt.figure(2)
plt.savefig(os.path.join(log_dir, 'fig_training_eplen.png'), dpi=200, bbox_inches='tight')
plt.show() 

# reconstruct env 
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

# obs, info = env.reset()
obs = env.reset()
dt_env = env.get_attr("_time_step")[0] * env.get_attr("_action_repeat")[0]
episode_reward = 0.0
ep_rewards = []
ep_energy = 0.0
start_pos = None
mass = None
ep_steps = 0
contact_hist = []

# initialize arrays to save data from simulation 
des_hist = []
vel_hist = []
wz_hist = []
rew_hist = []
ep_rew_components = []  # accumulate reward components for the current episode

for i in range(2000):
    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test if the outputs make sense)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards
    des_cmd = env.envs[0].env._desired_velocity.copy()  # [v_des_x, v_des_y, w_des_z]
    base_vel = info[0].get('base_vel')
    base_ang_vel = info[0].get('base_ang_vel')
    rew_comp = info[0].get('reward_components')
    des_hist.append(des_cmd)
    vel_hist.append(base_vel)
    wz_hist.append(base_ang_vel[2] if base_ang_vel is not None else None)
    rew_hist.append(rew_comp)
    if rew_comp is not None:
        ep_rew_components.append(rew_comp)
    # recover per-step energy from reward components if available
    step_energy = 0.0
    if rew_comp is not None and 'energy' in rew_comp:
        step_energy = -rew_comp['energy'] / 0.01  # undo scaling to get actual energy term
    ep_energy += step_energy
    if start_pos is None:
        start_pos = info[0]['base_pos']
    if mass is None:
        try:
            mass = float(np.sum(env.envs[0].env.robot._total_mass_urdf))
        except Exception:
            mass = None
    # log contacts for duty cycle/step time
    try:
        _, _, _, foot_boolean = env.envs[0].env.robot.GetContactInfo()
        contact_hist.append(np.array(foot_boolean, dtype=int))
    except Exception:
        # if contact info not available, skip this step
        contact_hist.append(None)
    ep_steps += 1
    
    if dones:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'[{ts}] episode_reward', episode_reward)
        print('Final base position', info[0]['base_pos'])
        print('Final base velocity', info[0].get('base_vel'))
        # print reward composition for this episode (sum over steps)
        if ep_rew_components:
            comp_keys = ep_rew_components[0].keys()
            totals = {k: sum(step[k] for step in ep_rew_components) for k in comp_keys}
            print('Episode reward breakdown:', {k: float(f"{v:.4f}") for k, v in totals.items()})
        ep_rew_components = []
        ep_rewards.append(float(episode_reward))
        # compute CoT and avg speed
        end_pos = np.array(info[0]['base_pos'])
        dist = 0.0 if start_pos is None else np.linalg.norm(end_pos[:2] - np.array(start_pos[:2]))
        ep_time = ep_steps * dt_env
        avg_speed = dist / ep_time if ep_time > 0 else 0.0
        cot = None
        if mass is not None and dist > 1e-6:
            cot = ep_energy / (mass * 9.81 * dist)
        # duty cycle and step duration
        duty_cycle_mean = None
        step_duration_mean = None
        valid_contacts = [c for c in contact_hist if c is not None]
        if valid_contacts:
            contacts = np.vstack(valid_contacts)  # shape (steps, 4)
            duty_cycle_per_leg = contacts.mean(axis=0)
            duty_cycle_mean = float(duty_cycle_per_leg.mean())
            step_durations = []
            for leg in range(contacts.shape[1]):
                transitions = np.sum(contacts[1:, leg] != contacts[:-1, leg])
                cycles = max(1, transitions / 2)
                step_durations.append(ep_time / cycles if ep_time > 0 else 0.0)
            step_duration_mean = float(np.mean(step_durations))
        print(f"Episode stats: dist={dist:.3f} m, avg_speed={avg_speed:.3f} m/s, CoT={cot if cot is not None else 'N/A'}, duty={duty_cycle_mean if duty_cycle_mean is not None else 'N/A'}, step_time={step_duration_mean if step_duration_mean is not None else 'N/A'} s")
        # reset episode accumulators
        ep_energy = 0.0
        start_pos = None
        ep_steps = 0
        contact_hist = []
        episode_reward = 0

    # [TODO] save data from current robot states for plots 
    # To get base position, for example: env.envs[0].env.robot.GetBasePosition() 
    
# [TODO] make plots

# visualize desired vs actual velocity tracking

des_hist = np.array(des_hist)
vel_hist = np.array(vel_hist)
wz_hist = np.array(wz_hist)
rew_hist = [r for r in rew_hist if r is not None]
# rew_hist is a list of reward component dicts

# if no episode finished during rollout, log the partial reward
if episode_reward != 0 and len(ep_rewards) == 0:
    ep_rewards.append(float(episode_reward))

dt = env.get_attr("_time_step")[0] * env.get_attr("_action_repeat")[0]
t = np.arange(len(des_hist)) * dt

plt.figure()
plt.plot(t, des_hist[:,0], label='v_des_x')
plt.plot(t, vel_hist[:,0], label='v_x')
plt.plot(t, des_hist[:,1], label='v_des_y')
plt.plot(t, vel_hist[:,1], label='v_y')
plt.plot(t, des_hist[:,2], label='w_des_z')
plt.plot(t, wz_hist, label='w_z')
plt.xlabel('time [s]')
plt.ylabel('velocity')
plt.legend()
plt.title('Desired vs Actual Velocity')
plt.savefig(os.path.join(log_dir, 'fig_vel_vs_des.png'), dpi=200, bbox_inches='tight')

plt.figure()
plt.plot(t, des_hist[:,0]-vel_hist[:,0], label='vx error')
plt.plot(t, des_hist[:,1]-vel_hist[:,1], label='vy error')
plt.plot(t, des_hist[:,2]-wz_hist, label='wz error')
plt.xlabel('time [s]')
plt.ylabel('error')
plt.legend()
plt.title('Velocity Tracking Error')
plt.savefig(os.path.join(log_dir, 'fig_vel_error.png'), dpi=200, bbox_inches='tight')

# reward breakdown plots (if available)
comp_hist = [(i, r) for i, r in enumerate(rew_hist) if r is not None]
if comp_hist:
    idxs, comps = zip(*comp_hist)
    comp_t = np.array([t[i] for i in idxs])
    keys = list(comps[0].keys())  # auto-detect components
    plt.figure()
    for k in keys:
        series = np.array([r[k] for r in comps])
        plt.plot(comp_t, series, label=k)
    plt.xlabel('time [s]')
    plt.ylabel('reward contribution per step')
    plt.legend()
    plt.title('Reward Components')
    plt.savefig(os.path.join(log_dir, 'fig_reward_components.png'), dpi=200, bbox_inches='tight')

# episode reward plot

plt.show()
