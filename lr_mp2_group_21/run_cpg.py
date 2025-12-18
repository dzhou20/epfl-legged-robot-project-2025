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
import matplotlib.colors as mcolors

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
leg_names = ["FR", "FL", "RR", "RL"]

env = QuadrupedGymEnv(render=True,              # visualize
                    on_rack=False,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    record_video=True
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
base_pos = np.zeros((TEST_STEPS, 3))
base_vel = np.zeros((TEST_STEPS, 3))
feet_contacts = np.zeros((4, TEST_STEPS), dtype=int)

#desired state
joint_pos_des = np.zeros((12, TEST_STEPS))
foot_pos_des = np.zeros((4, 3, TEST_STEPS))

mechanical_work = 0.0

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
  q = np.asarray(env.robot.GetMotorAngles())
  dq = np.asarray(env.robot.GetMotorVelocities())

  # loop through desired foot positions and calculate torques
  for i in range(4):
    # initialize torques for legi
    tau = np.zeros(3)

    # get desired foot i pos (xi, yi, zi) in leg frame
    leg_xyz = np.array([xs[i], sideSign[i] * foot_y, zs[i]])

    # log desired feet position for plotting
    foot_pos_des[i, :, j] = leg_xyz

    # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
    leg_q = env.robot.ComputeInverseKinematics(legID=i, xyz_coord=leg_xyz)  # [TODO: done by david]
    joint_pos_des[3*i:3*i+3, j] = leg_q
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

  base_pos[j, :] = env.robot.GetBasePosition()
  base_vel[j, :] = env.robot.GetBaseLinearVelocity()
  _, _, _, contact_bool = env.robot.GetContactInfo()
  feet_contacts[:, j] = np.asarray(contact_bool)
  mechanical_work += np.abs(np.dot(action, dq)) * TIME_STEP

  # send torques to robot and simulate TIME_STEP seconds 
  env.step(action) 
  dt = TIME_STEP
  cpg_r_dot = np.gradient(cpg_r, dt, axis=1)
  theta_unwrapped = np.unwrap(cpg_theta, axis=1)
  cpg_theta_dot = np.gradient(theta_unwrapped, dt, axis=1)


  # [TODO, done by david] save any CPG or robot states
  cpg_r[:, j] = cpg.get_r()
  cpg_theta[:, j] = cpg.get_theta()
  joint_pos[:, j] = q
  joint_vel[:, j] = dq
  for i in range(4):
    _, p_curr = env.robot.ComputeJacobianAndPosition(i, specific_q=q[3*i:3*i+3])
    foot_pos[i, :, j] = p_curr

final_base_pos = np.asarray(env.robot.GetBasePosition())

def compute_contact_stats(contact_signal, dt):
  signal = np.asarray(contact_signal, dtype=int)
  if signal.size == 0:
    return 0.0, 0.0, 0.0, 0.0

  stance_durations = []
  swing_durations = []
  current_state = signal[0]
  duration = 1

  for val in signal[1:]:
    if val == current_state:
      duration += 1
    else:
      if current_state == 1:
        stance_durations.append(duration * dt)
      else:
        swing_durations.append(duration * dt)
      current_state = val
      duration = 1

  if duration > 0:
    if current_state == 1:
      stance_durations.append(duration * dt)
    else:
      swing_durations.append(duration * dt)

  avg_stance = float(np.mean(stance_durations)) if stance_durations else 0.0
  avg_swing = float(np.mean(swing_durations)) if swing_durations else 0.0
  avg_step = avg_stance + avg_swing if (avg_stance > 0 and avg_swing > 0) else 0.0
  duty_cycle = float(np.mean(signal))
  return duty_cycle, avg_stance, avg_swing, avg_step

avg_body_velocity = float(np.mean(base_vel[:, 0]))
distance_traveled = float(final_base_pos[0] - base_pos[0, 0])
distance_mag = max(abs(distance_traveled), 1e-6)
robot_mass = float(np.sum(env.robot.GetTotalMassFromURDF()))
weight = robot_mass * 9.81
cot = mechanical_work / (weight * distance_mag) if weight > 0 else 0.0

print("\n===== Gait + Energy Metrics =====")
print(f"Average body velocity (x): {avg_body_velocity:.3f} m/s")
print(f"Distance traveled (x): {distance_traveled:.3f} m")
for i, name in enumerate(leg_names):
  duty, stance_dur, swing_dur, step_dur = compute_contact_stats(feet_contacts[i], TIME_STEP)
  print(f"{name}: duty={duty:.2f}, stance={stance_dur:.3f}s, swing={swing_dur:.3f}s, step={step_dur:.3f}s")
print(f"Cost of Transport (CoT): {cot:.3f}")

##################################################### 
# PLOTS
#####################################################
# Plot CPG amplitudes and phases
def darken_color(color, factor=0.6):
    """
    factor < 1  → darker
    factor = 1  → same
    """
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb * factor)

fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

# --- r ---
for i, name in enumerate(leg_names):
    axs[0, 0].plot(t, cpg_r[i, :], label=name)
axs[0, 0].set_title("CPG Amplitude r")
axs[0, 0].set_ylabel("r")
axs[0, 0].legend()
axs[0, 0].grid(True)

# --- r_dot ---
for i, name in enumerate(leg_names):
    axs[0, 1].plot(t, cpg_r_dot[i, :], label=name)
axs[0, 1].set_title("CPG Amplitude Derivative $\dot r$")
axs[0, 1].set_ylabel("$\dot r$")
axs[0, 1].grid(True)

# --- theta ---
for i, name in enumerate(leg_names):
    axs[1, 0].plot(t, cpg_theta[i, :], label=name)
axs[1, 0].set_title("CPG Phase $\\theta$")
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("$\\theta$ (rad)")
axs[1, 0].grid(True)

# --- theta_dot ---
for i, name in enumerate(leg_names):
    axs[1, 1].plot(t, cpg_theta_dot[i, :], label=name)
axs[1, 1].set_title("CPG Phase Derivative $\\dot \\theta$")
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_ylabel("$\\dot \\theta$ (rad/s)")
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()


# Plot joint positions for one leg (FR hip/thigh/calf)
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 3))

# actual joints
l1, = ax2.plot(t, joint_pos[0, :], label="FR hip")
l2, = ax2.plot(t, joint_pos[1, :], label="FR thigh")
l3, = ax2.plot(t, joint_pos[2, :], label="FR calf")

# dark colors for desired
c1 = darken_color(l1.get_color(), 0.6)
c2 = darken_color(l2.get_color(), 0.6)
c3 = darken_color(l3.get_color(), 0.6)

# desired joints (same color, darker)
ax2.plot(t, joint_pos_des[0, :], color=c1, linewidth=1.5, label="FR hip (desired)")
ax2.plot(t, joint_pos_des[1, :], color=c2, linewidth=1.5, label="FR thigh (desired)")
ax2.plot(t, joint_pos_des[2, :], color=c3, linewidth=1.5, label="FR calf (desired)")

ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Joint angle (rad)")
ax2.legend(ncol=2)
ax2.grid(True)

plt.show()


# Plot foot positions for the same leg (FR)
fig3, ax3 = plt.subplots(1, 1, figsize=(10, 3))

# actual foot xyz
lfx, = ax3.plot(t, foot_pos[0, 0, :], label="FR x")
lfy, = ax3.plot(t, foot_pos[0, 1, :], label="FR y")
lfz, = ax3.plot(t, foot_pos[0, 2, :], label="FR z")

# darker colors for desired trajectories
cfx = darken_color(lfx.get_color(), 0.6)
cfy = darken_color(lfy.get_color(), 0.6)
cfz = darken_color(lfz.get_color(), 0.6)

ax3.plot(t, foot_pos_des[0, 0, :], color=cfx, linewidth=1.5, label="FR x (desired)")
ax3.plot(t, foot_pos_des[0, 1, :], color=cfy, linewidth=1.5, label="FR y (desired)")
ax3.plot(t, foot_pos_des[0, 2, :], color=cfz, linewidth=1.5, label="FR z (desired)")

ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Foot position (m)")
ax3.legend(ncol=2)
ax3.grid(True)

plt.show()


# FR leg trajectory in leg frame (x vs z)
fig4, ax4 = plt.subplots(1, 1, figsize=(5, 5))

# use same color palette for consistency
leg_color = lfx.get_color()
leg_color_des = darken_color(leg_color, 0.6)

ax4.plot(foot_pos[0, 0, :], foot_pos[0, 2, :], label="FR foot", color=leg_color)
ax4.plot(foot_pos_des[0, 0, :], foot_pos_des[0, 2, :], label="FR foot (desired)", color=leg_color_des, linewidth=1.5)

ax4.set_xlabel("x (m)")
ax4.set_ylabel("z (m)")
ax4.set_title("FR Foot Trajectory in Leg Frame")
ax4.legend()
ax4.grid(True)
ax4.set_aspect('equal', adjustable='box')

plt.show()
