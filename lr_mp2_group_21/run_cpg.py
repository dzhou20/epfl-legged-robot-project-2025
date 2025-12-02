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

""" Run CPG """

import time
import numpy as np
import matplotlib

# adapt as needed for your system
from sys import platform
if platform =="darwin":
  matplotlib.use("Qt5Agg")
else:
  matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv

ADD_CARTESIAN_PD = True
TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length 
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(render=True,              # visualize
                    on_rack=False,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    # record_video=True
                    )

# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP)

TEST_STEPS = int(10 / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP

# [TODO, doned by david] initialize data structures to save CPG and robot states
# CPG status logs
cpg_r = np.zeros((4, TEST_STEPS))
cpg_theta = np.zeros((4, TEST_STEPS))

# robot status logs
joint_pos = np.zeros((12, TEST_STEPS))
joint_vel = np.zeros((12, TEST_STEPS))
foot_pos = np.zeros((4, 3, TEST_STEPS))

############## Sample Gains
# joint PD gains
kp=np.array([100,100,100])
kd=np.array([2,2,2])

# Cartesian PD gains
kpCartesian = np.diag([500]*3)
kdCartesian = np.diag([20]*3)

for j in range(TEST_STEPS):
  # initialize torque array to send to motors
  action = np.zeros(12) 

  # get desired foot positions from CPG 
  xs,zs = cpg.update()

  # [TODO, done by david] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
  q = env.robot.GetMotorAngles()
  dq = env.robot.GetMotorVelocities()

  # loop through desired foot positions and calculate torques
  for i in range(4):
    # initialize torques for legi
    tau = np.zeros(3)

    # get desired foot i pos (xi, yi, zi) in leg frame
    leg_xyz = np.array([xs[i], sideSign[i] * foot_y, zs[i]])

    # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
    leg_q = env.robot.ComputeInverseKinematics(legID=i, xyz_coord=leg_xyz)  # [TODO: done by david]

    # Add joint PD contribution to tau for leg i (Equation 4)
    # leg_q: desired joint angles for leg i from IK
    tau += kp * (leg_q - q[3*i:3*i+3]) + kd * (0 - dq[3*i:3*i+3])  # [TODO: done by david, maybe not exactly this way]

    # add Cartesian PD contribution
    if ADD_CARTESIAN_PD:
      # Get desired xyz position in leg frame (use ComputeJacobianAndPosition with the joint angles you just found above)
      J_des, p_des = env.robot.ComputeJacobianAndPosition(i, specific_q=leg_q)

      # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
      J_curr, p_curr = env.robot.ComputeJacobianAndPosition(i, specific_q=q[3*i:3*i+3])

      # Get current foot velocity in leg frame (Equation 2)
      v_curr = J_curr @ dq[3*i:3*i+3]  # [TODO: done by david] 

      # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
      v_des = np.zeros(3)  # if no explicit desired foot velocity
      tau += J_curr.T @ (kpCartesian @ (p_des - p_curr) + kdCartesian @ (v_des - v_curr))  # []

    # Set tau for legi in action vector
    action[3*i:3*i+3] = tau

  # send torques to robot and simulate TIME_STEP seconds 
  env.step(action) 

  # [TODO, done by david] save any CPG or robot states
  cpg_r[:, j] = cpg.get_r()
  cpg_theta[:, j] = cpg.get_theta()
  joint_pos[:, j] = q
  joint_vel[:, j] = dq
  for i in range(4):
    _, p_curr = env.robot.ComputeJacobianAndPosition(i, specific_q=q[3*i:3*i+3])
    foot_pos[i, :, j] = p_curr

##################################################### 
# PLOTS
#####################################################
# Plot CPG amplitudes and phases
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for i, name in enumerate(["FR", "FL", "RR", "RL"]):
  axs[0].plot(t, cpg_r[i, :], label=f"{name} r")
  axs[1].plot(t, cpg_theta[i, :], label=f"{name} theta")
axs[0].set_ylabel("Amplitude r")
axs[1].set_ylabel("Phase theta (rad)")
axs[1].set_xlabel("Time (s)")
axs[0].legend()
axs[1].legend()
axs[0].grid(True)
axs[1].grid(True)

# Plot joint positions for one leg (FR hip/thigh/calf)
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 3))
ax2.plot(t, joint_pos[0, :], label="FR hip")
ax2.plot(t, joint_pos[1, :], label="FR thigh")
ax2.plot(t, joint_pos[2, :], label="FR calf")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Joint angle (rad)")
ax2.legend()
ax2.grid(True)

plt.show()
