import optuna
import numpy as np
from functools import partial
from optuna.trial import Trial
from env.simulation import QuadSimulator, SimulationOptions
import argparse

from profiles import FootForceProfile

from quadruped_jump import (
    nominal_position,
    gravity_compensation,
    apply_force_profile,
    virtual_model,
)


N_LEGS = 4
N_JOINTS = 3


def quadruped_jump_optimization(objective_choice: str = "height"):
    # Initialize simulation
    # Feel free to change these options! (except for control_mode and timestep)
    sim_options = SimulationOptions(
        on_rack=False,  # Whether to suspend the robot in the air (helpful for debugging)
        render=False,  # Whether to use the GUI visualizer (slower than running in the background)
        # for optimization, use render=False to speed up! You can replay the best trial after optimization.
        record_video=False,  # Whether to record a video to file (needs render=True
        tracking_camera=True,  # Whether the camera follows the robot (instead of free)
    )
    simulator = QuadSimulator(sim_options)

    # Create a maximization problem
    objective = partial(evaluate_jumping, simulator=simulator, objective_choice=objective_choice)
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        study_name="Quadruped Jumping Optimization",
        sampler=sampler,
        direction="maximize",
    )

    # To set initial parameters for the first trial, you can enqueue a trial like this before calling study.optimize():
    initial_params = {
        "Fz": 225.0, "Fx": 75.0,
        "f0": 1.25,
    }
    study.enqueue_trial(initial_params)

    # Run the optimization
    # You can change the number of trials here (the maximum number of trials is 50)
    study.optimize(objective, n_trials=50, n_jobs=1)

    # Close the simulation
    simulator.close()

    # Log the results
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
    last_trial = study.trials[-1]
    print("Last trial value:", last_trial.value)
    print("Last trial params:", last_trial.params)

    # Plot results if objective is first_jump_distance
    if objective_choice == "first_jump_distance":
        trials = study.get_trials()
        successful_trials = [t for t in trials if t.value is not None]
        if successful_trials:
            import matplotlib.pyplot as plt
            trial_numbers = [t.number for t in successful_trials]
            distances = [t.value for t in successful_trials]
            plt.figure(figsize=(10, 6))
            plt.plot(trial_numbers, distances, marker='o', linestyle='-')
            plt.xlabel("Trial Number")
            plt.ylabel("Forward Jump Distance (m)")
            plt.title("Forward Jump Distance vs. Trial Number")
            plt.grid(True)
            plt.savefig("jump_distance_vs_trials.png")
            plt.close()

    # Plot optimized parameters vs. trial number
    trials = study.get_trials()
    successful_trials = [t for t in trials if t.value is not None]
    if successful_trials:
        import matplotlib.pyplot as plt
        trial_numbers = [t.number for t in successful_trials]
        
        # Extract parameters and objective values
        fx_values = [t.params["Fx"] for t in successful_trials]
        fz_values = [t.params["Fz"] for t in successful_trials]
        f0_values = [t.params["f0"] for t in successful_trials]
        objective_values = [t.value for t in successful_trials]

        fig, axs = plt.subplots(4, 1, figsize=(10, 20), sharex=True)

        # Plot Fx
        axs[0].plot(trial_numbers, fx_values, marker='o', linestyle='-')
        axs[0].set_ylabel("Fx")
        axs[0].set_title("Optimized Parameters and Objective Value vs. Trial Number")
        axs[0].grid(True)

        # Plot Fz
        axs[1].plot(trial_numbers, fz_values, marker='o', linestyle='-', color='r')
        axs[1].set_ylabel("Fz")
        axs[1].grid(True)

        # Plot f0
        axs[2].plot(trial_numbers, f0_values, marker='o', linestyle='-', color='g')
        axs[2].set_ylabel("f0")
        axs[2].grid(True)

        # Plot Objective Value
        axs[3].plot(trial_numbers, objective_values, marker='o', linestyle='-', color='purple')
        axs[3].set_ylabel("Objective Value")
        axs[3].set_xlabel("Trial Number")
        axs[3].grid(True)

        plt.tight_layout()
        plt.savefig("optimized_params_vs_trials.png")
        plt.close()

    # Replay and record the best and last jumps
    print("\nReplaying and recording the best jump...")
    best_video_path = "/home/dzhou20/epfl/legged_robots/epfl-legged-robot-project-2025/lr_mp1_group_21/videos/best_jump.mp4"
    best_replay_options = SimulationOptions(render=True, on_rack=False, tracking_camera=True, record_video=best_video_path)
    best_replay_simulator = QuadSimulator(best_replay_options)
    replay_jump(best_replay_simulator, study.best_params)
    best_replay_simulator.close()
    print(f"Best jump video saved to {best_video_path}")

    print("\nReplaying and recording the last jump...")
    last_video_path = "/home/dzhou20/epfl/legged_robots/epfl-legged-robot-project-2025/lr_mp1_group_21/videos/last_jump.mp4"
    last_replay_options = SimulationOptions(render=True, on_rack=False, tracking_camera=True, record_video=last_video_path)
    last_replay_simulator = QuadSimulator(last_replay_options)
    replay_jump(last_replay_simulator, last_trial.params)
    last_replay_simulator.close()
    print(f"Last jump video saved to {last_video_path}")


    # OPTIONAL: add additional functions here (e.g., plotting, recording to file)
    # E.g., cycling through all the evaluated parameters and values:
    for trial in study.get_trials():
        trial.number  # Number of the trial
        trial.params  # Used parameters
        trial.value  # Resulting objective function value


def evaluate_jumping(trial: Trial, simulator: QuadSimulator, objective_choice: str) -> float:
    # the number of four legs: FR(Front Right), FL(Front Left), RR(Rear Right), RL(Rear Left)
    # Parameters for all legs
    Fz = trial.suggest_float("Fz", 150, 350)
    Fx = trial.suggest_float("Fx", 0, 150)
    Fy = 0.0  # To avoid sideways movement, set Fy to 0
    f0 = trial.suggest_float("f0", 1, 1.75)
    
    f1 = 0.5 # This seems to be a constant

    # Reset the simulation
    simulator.reset()

    # Extract simulation options
    sim_options = simulator.options

    # Determine number of jumps to simulate
    if objective_choice == "first_jump_distance":
        n_jumps = 1  # Simulate for long enough to capture one full jump
    else:
        n_jumps = 5 # Feel free to change this number
    
    # Using average of f0 for jump duration calculation
    f0_avg = f0
    jump_duration = 1/(2*f0_avg) + 1/(2*f1)
    n_steps = int(n_jumps * jump_duration / sim_options.timestep)

    # Create separate profiles for each leg
    profile_FR = FootForceProfile(f0=f0, f1=f1, Fx=Fx, Fy=Fy, Fz=Fz)
    profile_FL = FootForceProfile(f0=f0, f1=f1, Fx=Fx, Fy=Fy, Fz=Fz)
    profile_RR = FootForceProfile(f0=f0, f1=f1, Fx=Fx, Fy=Fy, Fz=Fz)
    profile_RL = FootForceProfile(f0=f0, f1=f1, Fx=Fx, Fy=Fy, Fz=Fz)

    # The order of legs is FR, FL, RR, RL
    force_profiles = [profile_FR, profile_FL, profile_RR, profile_RL]
    
    base_position_history = []
    contact_history = []
    base_row_pitch_yaw_history = []
    for _ in range(n_steps):
        # Step the oscillator
        for profile in force_profiles:
            profile.step(sim_options.timestep)

        # Compute torques as motor targets (reuses your controller functions)
        tau = np.zeros(N_JOINTS * N_LEGS)
        tau += nominal_position(simulator)
        tau += apply_force_profile(simulator, force_profiles)
        tau += gravity_compensation(simulator)

        # If touching the ground, add virtual model
        foot_contacts = simulator.get_foot_contacts()
        on_ground = any(foot_contacts)
        if on_ground:
            tau += virtual_model(simulator)

        # Set the motor commands and step the simulation
        simulator.set_motor_targets(tau)
        simulator.step()
        base_position_history.append(simulator.get_base_position())
        contact_history.append(foot_contacts)
        base_row_pitch_yaw_history.append(simulator.get_base_orientation_roll_pitch_yaw())

    base_position_history = np.array(base_position_history)
    base_row_pitch_yaw_history = np.array(base_row_pitch_yaw_history)

    def penalty_roll_pitch(base_row_pitch_yaw_history: np.ndarray) -> float:
        """Calculate penalty based on roll and pitch deviations from upright position."""
        roll = base_row_pitch_yaw_history[:, 0]  
        pitch = base_row_pitch_yaw_history[:, 1]
        for roll_angle in roll:
            if abs(roll_angle) > np.pi/2:  # 30 degrees in radians
                return -100.0  # Large penalty for excessive roll
        for pitch_angle in pitch:
            if abs(pitch_angle) > np.pi/2:  # 30 degrees in radians
                return -100.0  # Large penalty for excessive pitch
        return 0.0  # No penalty if within limits
        

    def get_max_height(history: np.ndarray) -> float:
        """Get the maximum height reached by the base during the simulation."""
        max_height = np.max(history[:, 2])  # Z coordinate is the height
        return max_height

    def get_max_distance(history: np.ndarray) -> float:
        """Get the maximum distance covered by the base during the simulation."""
        initial_pos = history[0]
        final_pos = history[-1]
        max_dist_x = final_pos[0] - initial_pos[0]
        max_dist_y = final_pos[1] - initial_pos[1]
        max_dist_z = final_pos[2] - initial_pos[2]
        max_dist = (max_dist_x)
        if penalty_roll_pitch(base_row_pitch_yaw_history) < 0:
            return 0.0
        return max_dist

    def get_first_jump_distance(history: np.ndarray, contacts_hist: list) -> float:
        """
        Calculates distance from start to the first stable landing, ensuring a jump occurred.
        """
        # Find the first index of an air phase (a jump)
        first_air_phase_idx = -1
        for i, contacts in enumerate(contacts_hist):
            if not np.any(contacts):
                first_air_phase_idx = i
                print("contacts:", contacts)
                break
        print("first_air_phase_idx:", first_air_phase_idx)

        if first_air_phase_idx == -1:
            print("Debug: No air phase detected.")
            return 0.0 # No jump occurred

        # Find the first stable landing (all four feet on ground) after the jump started
        stable_landing_idx = -1
        for i in range(first_air_phase_idx, len(contacts_hist)):
            if all(contacts_hist[i]):
                stable_landing_idx = i
                break
        
        if stable_landing_idx == -1:
            print("Debug: No stable four-foot landing detected after air phase.")
            return 0.0

        start_pos = history[0]
        landing_pos = history[stable_landing_idx]

        distance_x = landing_pos[0] - start_pos[0]
        distance_y = landing_pos[1] - start_pos[1]

        print(f"Debug: Start Pos: {start_pos}, Landing Pos: {landing_pos}")
        print(f"Debug: Landing Index: {stable_landing_idx}")
        print(f"Debug: X Distance: {distance_x:.4f} m, Y Distance: {distance_y:.4f} m")
        
        # The optimization objective is the forward distance
        return distance_x

    if objective_choice == "height":
        objective_value = get_max_height(base_position_history)
        print(f"Trial {trial.number}: max height = {objective_value:.4f} m")
    elif objective_choice == "distance":
        objective_value = get_max_distance(base_position_history)
        print(f"Trial {trial.number}: max distance = {objective_value:.4f} m")
    elif objective_choice == "first_jump_distance":
        objective_value = get_first_jump_distance(base_position_history, contact_history)
        print(f"Trial {trial.number}: first jump distance = {objective_value:.4f} m")
    else:
        raise ValueError(f"Invalid objective choice: {objective_choice}")
    
    return objective_value


def replay_jump(simulator: QuadSimulator, params: dict):
    """
    Simulates a jump with the given parameters.
    This function is used to replay the best jump found during optimization.
    """
    # Extract parameters
    Fz = params["Fz"]
    Fx = params["Fx"]
    Fy = 0.0  # To avoid sideways movement, set Fy to 0
    f0 = params["f0"]
    
    f1 = 0.5 # This seems to be a constant

    # Reset the simulation
    simulator.reset()

    # Extract simulation options
    sim_options = simulator.options

    # Determine number of jumps to simulate
    n_jumps = 5 # Simulate for long enough to capture one full jump
    
    # Using average of f0 for jump duration calculation
    f0_avg = f0
    jump_duration = 1/(2*f0_avg) + 1/(2*f1)
    n_steps = int(n_jumps * jump_duration / sim_options.timestep)

    # Create separate profiles for each leg
    profile_FR = FootForceProfile(f0=f0, f1=f1, Fx=Fx, Fy=Fy, Fz=Fz)
    profile_FL = FootForceProfile(f0=f0, f1=f1, Fx=Fx, Fy=Fy, Fz=Fz)
    profile_RR = FootForceProfile(f0=f0, f1=f1, Fx=Fx, Fy=Fy, Fz=Fz)
    profile_RL = FootForceProfile(f0=f0, f1=f1, Fx=Fx, Fy=Fy, Fz=Fz)

    # The order of legs is FR, FL, RR, RL
    force_profiles = [profile_FR, profile_FL, profile_RR, profile_RL]
    
    for _ in range(n_steps):
        # Step the oscillator
        for profile in force_profiles:
            profile.step(sim_options.timestep)

        # Compute torques as motor targets (reuses your controller functions)
        tau = np.zeros(N_JOINTS * N_LEGS)
        tau += nominal_position(simulator)
        tau += apply_force_profile(simulator, force_profiles)
        tau += gravity_compensation(simulator)

        # If touching the ground, add virtual model
        foot_contacts = simulator.get_foot_contacts()
        on_ground = any(foot_contacts)
        if on_ground:
            tau += virtual_model(simulator)

        # Set the motor commands and step the simulation
        simulator.set_motor_targets(tau)
        simulator.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quadruped Jumping Optimization")
    parser.add_argument(
        "--objective",
        type=str,
        default="height",
        choices=["height", "distance", "first_jump_distance"],
        help="Objective function to optimize: 'height', 'distance', or 'first_jump_distance'.",
    )
    args = parser.parse_args()

    quadruped_jump_optimization(objective_choice=args.objective)
