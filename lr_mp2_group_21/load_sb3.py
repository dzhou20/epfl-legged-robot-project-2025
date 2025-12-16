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
log_dir = interm_dir + '121625162359'

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
env_config = {
    "motor_control_mode":"CPG",  # PD is default, CPG for our project
    "task_env": "LR_COURSE_TASK",
    "observation_space_mode": "LR_COURSE_OBS",
    # "observation_space_mode": "FWD_LOCOMOTION",
    "add_noise": False,
    # add_noise will be set to False to make policy learning faster
}
env_config['render'] = True
env_config['record_video'] = False
env_config['add_noise'] = False 

# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
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
episode_reward = 0.0

# initialize arrays to save data from simulation 
des_hist = []
vel_hist = []
wz_hist = []

for i in range(2000):
    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test if the outputs make sense)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards

    # log desired and actual velocities
    des_cmd = env.envs[0].env._desired_velocity.copy()  # [v_des_x, v_des_y, w_des_z]
    base_vel = info[0].get('base_vel')
    base_ang_vel = info[0].get('base_ang_vel')
    des_hist.append(des_cmd)
    vel_hist.append(base_vel)
    wz_hist.append(base_ang_vel[2] if base_ang_vel is not None else None)
    
    if dones:
        print('episode_reward', episode_reward)
        print('Final base position', info[0]['base_pos'])
        print('Final base velocity', info[0].get('base_vel'))
        episode_reward = 0

    # [TODO] save data from current robot states for plots 
    # To get base position, for example: env.envs[0].env.robot.GetBasePosition() 
    
# [TODO] make plots

# visualize desired vs actual velocity tracking

des_hist = np.array(des_hist)
vel_hist = np.array(vel_hist)
wz_hist = np.array(wz_hist)

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

plt.figure()
plt.plot(t, des_hist[:,0]-vel_hist[:,0], label='vx error')
plt.plot(t, des_hist[:,1]-vel_hist[:,1], label='vy error')
plt.plot(t, des_hist[:,2]-wz_hist, label='wz error')
plt.xlabel('time [s]')
plt.ylabel('error')
plt.legend()
plt.title('Velocity Tracking Error')

plt.show()
