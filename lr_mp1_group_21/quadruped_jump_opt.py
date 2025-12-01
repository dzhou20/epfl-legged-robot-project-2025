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

    # To set initial parameters for the first trial, you can enqueue a trial like this
    if objective_choice == "distance":
        initial_params = {"Fx": 75.0, "Fz": 225.0, "f0": 1.25}
    elif objective_choice == "lateral_distance":
        initial_params = {"Fy": 75.0, "Fz": 225.0, "f0": 1.25}
    elif objective_choice == "twist":
        initial_params = {"Fy_front": 75.0, "Fz": 225.0, "f0": 1.25}
    
    if 'initial_params' in locals():
        study.enqueue_trial(initial_params)

    # Run the optimization
    # You can change the number of trials here (the maximum number of trials is 50)
    study.optimize(objective, n_trials=20, n_jobs=1)

    # Close the simulation
    simulator.close()

    # Log the results
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
    last_trial = study.trials[-1]
    print("Last trial value:", last_trial.value)
    print("Last trial params:", last_trial.params)


    # Plot optimized parameters vs. trial number
    trials = study.get_trials()
    successful_trials = [t for t in trials if t.value is not None]
    if successful_trials:
        import matplotlib.pyplot as plt
        trial_numbers = [t.number for t in successful_trials]
        objective_values = [t.value for t in successful_trials]

        # The parameters that were optimized
        optimized_params = list(successful_trials[0].params.keys())
        num_params = len(optimized_params)
        
        fig, axs = plt.subplots(num_params + 1, 1, figsize=(10, 5 * (num_params + 1)), sharex=True)
        fig.suptitle("Optimized Parameters and Objective Value vs. Trial Number")

        # Plot optimized parameters
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        for i, param_name in enumerate(optimized_params):
            param_values = [t.params[param_name] for t in successful_trials]
            axs[i].plot(trial_numbers, param_values, marker='o', linestyle='-', color=colors[i % len(colors)])
            axs[i].set_ylabel(param_name)
            axs[i].grid(True)

        # Plot Objective Value
        axs[num_params].plot(trial_numbers, objective_values, marker='o', linestyle='-', color='purple')
        axs[num_params].set_ylabel("Objective Value")
        axs[num_params].set_xlabel("Trial Number")
        axs[num_params].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to make room for suptitle
        plt.savefig("optimized_params_vs_trials.png")
        plt.close()

    # Replay and record the best jump
    print("\nReplaying and recording the best jump...")
    best_video_path = "/home/dzhou20/epfl/legged_robots/epfl-legged-robot-project-2025/lr_mp1_group_21/videos/best_jump.mp4"
    best_replay_options = SimulationOptions(render=True, on_rack=False, tracking_camera=True, record_video=best_video_path)
    best_replay_simulator = QuadSimulator(best_replay_options)
    replay_jump(best_replay_simulator, study.best_params, objective_choice)
    best_replay_simulator.close()
    print(f"Best jump video saved to {best_video_path}")


    # OPTIONAL: add additional functions here (e.g., plotting, recording to file)
    # E.g., cycling through all the evaluated parameters and values:
    for trial in study.get_trials():
        trial.number  # Number of the trial
        trial.params  # Used parameters
        trial.value  # Resulting objective function value


def evaluate_jumping(trial: Trial, simulator: QuadSimulator, objective_choice: str) -> float:
    # the number of four legs: FR(Front Right), FL(Front Left), RR(Rear Right), RL(Rear Left)
    # Common parameters
    f0 = trial.suggest_float("f0", 0.75, 1.75)
    f1 = 1.25 # This seems to be a constant

    # Objective-specific parameters and profiles
    if objective_choice == "distance":
        Fx = trial.suggest_float("Fx", 0, 150)
        Fz = trial.suggest_float("Fz", 150, 350)
        Fy = 0.0
        profile = FootForceProfile(f0=f0, f1=f1, Fx=Fx, Fy=Fy, Fz=Fz)
        force_profiles = [profile] * 4
    
    elif objective_choice == "lateral_distance":
        Fy = trial.suggest_float("Fy", -150, 150)
        Fz = trial.suggest_float("Fz", 150, 350)
        Fx = 0.0
        profile = FootForceProfile(f0=f0, f1=f1, Fx=Fx, Fy=Fy, Fz=Fz)
        force_profiles = [profile] * 4
    
    elif objective_choice == "twist":
        Fy_front = trial.suggest_float("Fy_front", 0, 150)
        Fz = trial.suggest_float("Fz", 150, 350)
        Fx = 0.0
        Fy_rear = -Fy_front
        profile_front = FootForceProfile(f0=f0, f1=f1, Fx=Fx, Fy=Fy_front, Fz=Fz)
        profile_rear = FootForceProfile(f0=f0, f1=f1, Fx=Fx, Fy=Fy_rear, Fz=Fz)
        force_profiles = [profile_front, profile_front, profile_rear, profile_rear]
    else:
        raise ValueError(f"Invalid objective choice for parameter suggestion: {objective_choice}")

    # Reset the simulation
    simulator.reset()

    # Extract simulation options
    sim_options = simulator.options

    # Determine number of jumps to simulate
    n_jumps = 5  # Simulate for 5 jumps

    # Using average of f0 for jump duration calculation
    f0_avg = f0
    jump_duration = 1/(2*f0_avg) + 1/(2*f1)
    n_steps = int(n_jumps * jump_duration / sim_options.timestep)

    # The order of legs is FR, FL, RR, RL

    
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
            if abs(roll_angle) > np.pi/3:  # 30 degrees in radians
                return -100.0  # Large penalty for excessive roll
        for pitch_angle in pitch:
            if abs(pitch_angle) > np.pi/3:  # 30 degrees in radians
                return -100.0  # Large penalty for excessive pitch
        return 0.0  # No penalty if within limits
        

    def get_max_distance(history: np.ndarray) -> float:
        """Get the maximum distance covered by the base during the simulation."""
        initial_pos = history[0]
        final_pos = history[-1]
        max_dist_x = final_pos[0] - initial_pos[0]
        if penalty_roll_pitch(base_row_pitch_yaw_history) < 0:
            return 0.0
        return max_dist_x

    def get_max_lateral_distance(history: np.ndarray) -> float:
        """Get the maximum lateral distance covered by the base during the simulation."""
        initial_pos = history[0]
        final_pos = history[-1]
        max_dist_y = final_pos[1] - initial_pos[1]
        if penalty_roll_pitch(base_row_pitch_yaw_history) < 0:
            return 0.0
        return abs(max_dist_y)

    def get_max_twist(rpy_history: np.ndarray) -> float:
        """Get the maximum twist (yaw change) of the base during the simulation."""
        initial_orientation = rpy_history[0]
        final_orientation = rpy_history[-1]
        yaw_initial = initial_orientation[2]
        yaw_final = final_orientation[2]
        if penalty_roll_pitch(base_row_pitch_yaw_history) < 0:
            return 0.0
        return abs(yaw_final - yaw_initial)

    

    if objective_choice == "distance":
        objective_value = get_max_distance(base_position_history)
        print(f"Trial {trial.number}: max distance = {objective_value:.4f} m")
    elif objective_choice == "lateral_distance":
        objective_value = get_max_lateral_distance(base_position_history)
        print(f"Trial {trial.number}: max lateral distance = {objective_value:.4f} m")
    elif objective_choice == "twist":
        objective_value = get_max_twist(base_row_pitch_yaw_history)
        print(f"Trial {trial.number}: max twist = {objective_value:.4f} rad")
    else:
        raise ValueError(f"Invalid objective choice: {objective_choice}")

    return objective_value


def replay_jump(simulator: QuadSimulator, params: dict, objective_choice: str):
    """
    Simulates a jump with the given parameters.
    This function is used to replay the best jump found during optimization.
    """
    # Common parameters
    f0 = params["f0"]
    f1 = 1.25 # This seems to be a constant

    # Objective-specific parameters and profiles
    if objective_choice == "distance":
        Fx = params["Fx"]
        Fz = params["Fz"]
        Fy = 0.0
        profile = FootForceProfile(f0=f0, f1=f1, Fx=Fx, Fy=Fy, Fz=Fz)
        force_profiles = [profile] * 4
    elif objective_choice == "lateral_distance":
        Fy = params["Fy"]
        Fz = params["Fz"]
        Fx = 0.0
        profile = FootForceProfile(f0=f0, f1=f1, Fx=Fx, Fy=Fy, Fz=Fz)
        force_profiles = [profile] * 4
    elif objective_choice == "twist":
        Fy_front = params["Fy_front"]
        Fz = params["Fz"]
        Fx = 0.0
        Fy_rear = -Fy_front
        profile_front = FootForceProfile(f0=f0, f1=f1, Fx=Fx, Fy=Fy_front, Fz=Fz)
        profile_rear = FootForceProfile(f0=f0, f1=f1, Fx=Fx, Fy=Fy_rear, Fz=Fz)
        force_profiles = [profile_front, profile_front, profile_rear, profile_rear]
    else:
        raise ValueError(f"Invalid objective choice for replay: {objective_choice}")

    # Reset the simulation
    simulator.reset()

    # Extract simulation options
    sim_options = simulator.options

    # Determine number of jumps to simulate
    n_jumps = 5 # Simulate for 5 jumps
    
    # Using average of f0 for jump duration calculation
    f0_avg = f0
    jump_duration = 1/(2*f0_avg) + 1/(2*f1)
    n_steps = int(n_jumps * jump_duration / sim_options.timestep)

    # The order of legs is FR, FL, RR, RL
    
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
        default="distance",
        choices=["distance", "lateral_distance", "twist"],
        help="Objective function to optimize: 'distance', 'lateral_distance', or 'twist'.",
    )
    args = parser.parse_args()

    quadruped_jump_optimization(objective_choice=args.objective)
