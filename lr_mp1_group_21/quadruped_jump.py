import numpy as np
from env.simulation import QuadSimulator, SimulationOptions

from profiles import FootForceProfile

N_LEGS = 4
N_JOINTS = 3


def quadruped_jump():
    # Initialize simulation
    # Feel free to change these options! (except for control_mode and timestep)
    sim_options = SimulationOptions(
        on_rack=False,  # Whether to suspend the robot in the air (helpful for debugging)
        # to work with the code, use on_rack=False
        render=True,  # Whether to use the GUI visualizer (slower than running in the background)
        record_video=False,  # Whether to record a video to file (needs render=True)
        tracking_camera=True,  # Whether the camera follows the robot (instead of free)
    )
    simulator = QuadSimulator(sim_options)

    # Determine number of jumps to simulate
    n_jumps = 10  # Feel free to change this number
    jump_duration = 0.8  # TODO: determine how long a jump takes
    n_steps = int(n_jumps * jump_duration / sim_options.timestep)

    # TODO: set parameters for the foot force profile here
    # We create a list of 4 force profiles, one for each leg.
    # This allows us to have different force profiles for each leg.
    # For now, they are all the same, but you can change them individually.
    force_profiles = [
        FootForceProfile(f0=1.25, f1=0.25, Fx=0, Fy=0, Fz=250), # for _ in range(N_LEGS)
        FootForceProfile(f0=1.25, f1=0.25, Fx=0, Fy=25, Fz=250),
        FootForceProfile(f0=1.25, f1=0.25, Fx=0, Fy=50, Fz=250),
        FootForceProfile(f0=1.25, f1=0.25, Fx=0, Fy=75, Fz=250)
    ]
    # f0 [0.75, 1.75]
    # Fx [0 150]
    # Fy [0 150]
    # Fz [150 350]

    for _ in range(n_steps):
        # If the simulator is closed, stop the loop
        if not simulator.is_connected():
            break

        # Step the oscillators
        for profile in force_profiles:
            profile.step(sim_options.timestep)

        # Compute torques as motor targets
        # The convention is as follows:
        # - A 1D array where the torques for the 3 motors follow each other for each leg
        # - The first 3 elements are the hip, thigh, calf torques for the FR leg.
        # - The order of the legs is FR, FL, RR, RL (front/rear,right/left)
        # - The resulting torque array is therefore structured as follows:
        # [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]
        tau = np.zeros(N_JOINTS * N_LEGS)

        # TODO: implement the functions below, and add potential controller parameters as function parameters here
        tau += nominal_position(simulator,Kp,Kd,Kd_point)
        tau += apply_force_profile(simulator, force_profiles)
        tau += gravity_compensation(simulator)

        # If touching the ground, add virtual model
        foot_contacts = simulator.get_foot_contacts()  # use contact for 4 foot
        on_ground = any(foot_contacts)  # True if any foot is touching the ground
        # print(foot_contacts)
        if on_ground:
            tau += virtual_model(simulator,Katt)
      
        # Set the motor commands and step the simulation
        simulator.set_motor_targets(tau)
        simulator.step()

    # Close the simulation
    simulator.close()

    # OPTIONAL: add additional functions here (e.g., plotting)

Kp = np.diag([400,400,400])
Kd = np.diag([8,8,8])
Kd_point = np.diag([0.8,0.8,0.8])
Katt = 200

def nominal_position(
    simulator: QuadSimulator,
    # OPTIONAL: add potential controller parameters here (e.g., gains)
    Kp = np.diag([400,400,400]),
    Kd = np.diag([8,8,8]),
    Kd_point = np.diag([0.8,0.8,0.8]),
    nominal_foot_pos = np.array([0,0,-0.25]),  
) -> np.ndarray:
    # All motor torques are in a single array
    tau = np.zeros(N_JOINTS * N_LEGS)
    for leg_id in range(N_LEGS):
        # #TODO: compute nominal position torques for leg_id
        J, ee_pos_legFrame =simulator.get_jacobian_and_position(leg_id)
        
        motor_vel = simulator.get_motor_velocities(leg_id)

        foot_linvel = J @ motor_vel

        # calculate torque
        tau_i= J.T @ ( Kp @ ( nominal_foot_pos - ee_pos_legFrame) + Kd @ (-foot_linvel) )+ Kd_point @(-motor_vel)

        # Store in torques array
        tau[leg_id * N_JOINTS : leg_id * N_JOINTS + N_JOINTS] = tau_i
    return tau


def virtual_model(
    simulator: QuadSimulator,
    Katt = 200,
    # OPTIONAL: add potential controller parameters here (e.g., gains)
) -> np.ndarray:
    # All motor torques are in a single array
    tau = np.zeros(N_JOINTS * N_LEGS)
    Corner_initial = np.array([[1,1,-1,-1],[-1,1,-1,1],[0,0,0,0]])
    R = simulator.get_base_orientation_matrix()
    P = R @ Corner_initial
    FVMC = np.zeros ((3,4))
    
    FVMC[-1,:]=Katt*(np.array([0,0,1])@P)
    
    
    for leg_id in range(N_LEGS):

        # #TODO: compute virtual model torques for leg_id

        J, ee_pos_legFrame =simulator.get_jacobian_and_position(leg_id)
        tau_i = J.T @ FVMC[:,leg_id]

        # Store in torques array
        tau[leg_id * N_JOINTS : leg_id * N_JOINTS + N_JOINTS] = tau_i

    return tau


def gravity_compensation(
    simulator: QuadSimulator,
    # OPTIONAL: add potential controller parameters here (e.g., gains)
) -> np.ndarray:
    # All motor torques are in a single array
    g = 9.81
    perleg_mass = simulator.get_mass()/4
    tau = np.zeros(N_JOINTS * N_LEGS)
    for leg_id in range(N_LEGS):
        J, ee_pos_legFrame =simulator.get_jacobian_and_position(leg_id)
        # TODO: compute gravity compensation torques for leg_id
        Fg = np.array([0, 0, -perleg_mass * g])
        tau_i = J.T @ Fg
        # Store in torques array
        tau[leg_id * N_JOINTS : leg_id * N_JOINTS + N_JOINTS] = tau_i

    return tau


def apply_force_profile(
    simulator: QuadSimulator,
    force_profiles: list[FootForceProfile],
    # OPTIONAL: add potential controller parameters here (e.g., gains)
) -> np.ndarray:
    # All motor torques are in a single array
    tau = np.zeros(N_JOINTS * N_LEGS)
    for leg_id in range(N_LEGS):

        # TODO: compute force profile torques for leg_id
        # tau_i = np.zeros(3)
        tau_i = simulator.get_jacobian_and_position(leg_id)[0].T @ force_profiles[leg_id].force()
        # Store in torques array
        tau[leg_id * N_JOINTS : leg_id * N_JOINTS + N_JOINTS] = tau_i

    return tau


if __name__ == "__main__":
    quadruped_jump()
