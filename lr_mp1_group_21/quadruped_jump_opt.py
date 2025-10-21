import optuna
import numpy as np
from functools import partial
from optuna.trial import Trial
from env.simulation import QuadSimulator, SimulationOptions
import argparse

from profiles import FootForceProfile

from quadruped_jump import (
    nominal_position_ground,
    nominal_position_fly,
    gravity_compensation,
    apply_force_profile,
    virtual_model,
    Kp,
    Kd,
    Kd_point,
    Katt,
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
    # sampler = optuna.samplers.TPESampler(seed=42)
    sampler = optuna.samplers.TPESampler(seed=52)
    study = optuna.create_study(
        study_name="Quadruped Jumping Optimization",
        sampler=sampler,
        direction="maximize",
    )

    # To set initial parameters for the first trial, you can enqueue a trial like this
    if objective_choice == "distance":
        initial_params = {"Fx": 75.0, "Fz": 225.0, "f0": 1.25}
    elif objective_choice == "lateral_distance":
        initial_params = {"Fy": 60.0, "Fz": 350.0, "f0": 1.25}
    elif objective_choice == "twist":
        initial_params = {"Fx": 75.0, "Fy": 75.0, "f0": 1.25}
    
    if 'initial_params' in locals():
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

        # Create 2x2 grid layout
        fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        fig.suptitle("Optimized Parameters and Objective Value vs. Trial Number")

        # Flatten the 2x2 array for easier indexing
        axs_flat = axs.flatten()

        # Plot optimized parameters
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        for i, param_name in enumerate(optimized_params):
            param_values = [t.params[param_name] for t in successful_trials]
            axs_flat[i].plot(trial_numbers, param_values, marker='o', linestyle='-', color=colors[i % len(colors)])
            axs_flat[i].set_ylabel(param_name)
            axs_flat[i].grid(True)
            # Add x-label to bottom row
            if i >= 2:
                axs_flat[i].set_xlabel("Trial Number")

        # Plot Objective Value in the last subplot
        axs_flat[num_params].plot(trial_numbers, objective_values, marker='o', linestyle='-', color='purple')
        axs_flat[num_params].set_ylabel("Objective Value")
        axs_flat[num_params].set_xlabel("Trial Number")
        axs_flat[num_params].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to make room for suptitle
        plt.savefig("optimized_params_vs_trials.png")
        plt.close()

    # Replay and record the best jump
    print("\nReplaying and recording the best jump...")
    print("Video will be automatically saved to the 'videos' directory with timestamp.")
    best_replay_options = SimulationOptions(render=True, on_rack=False, tracking_camera=True, record_video=True)
    best_replay_simulator = QuadSimulator(best_replay_options)
    replay_jump(best_replay_simulator, study.best_params, objective_choice)
    best_replay_simulator.close()
    print(f"Best jump replay completed. Check the 'videos' directory for the recording.")


    # OPTIONAL: add additional functions here (e.g., plotting, recording to file)
    # E.g., cycling through all the evaluated parameters and values:
    for trial in study.get_trials():
        trial.number  # Number of the trial
        trial.params  # Used parameters
        trial.value  # Resulting objective function value


def evaluate_jumping(trial: Trial, simulator: QuadSimulator, objective_choice: str) -> float:
    # the number of four legs: FR(Front Right), FL(Front Left), RR(Rear Right), RL(Rear Left)
    # Parameters are now conditional on the objective
    if objective_choice == "distance":
        para = 0
        Fx = trial.suggest_float("Fx", 0, 150)
        Fz = trial.suggest_float("Fz", 150 + para, 350 + para)
        f0 = trial.suggest_float("f0", 0.75, 1.75)
        Fy = 0.0
    elif objective_choice == "lateral_distance":
        Fy = trial.suggest_float("Fy", 30, 150)
        Fz = trial.suggest_float("Fz", 250, 450)
        f0 = trial.suggest_float("f0", 0.75, 1.75)
        Fx = -18.0
    elif objective_choice == "twist":
        Fx = trial.suggest_float("Fx", 0, 150)
        Fy = trial.suggest_float("Fy", 0, 150)
        f0 = trial.suggest_float("f0", 0.75, 1.75)
        Fz = 250.0  # Using a fixed, balanced value for Fz
    else:
        raise ValueError(f"Invalid objective choice for parameter suggestion: {objective_choice}")
    
    f1 = 0.8 # This seems to be a constant

    # Print trial parameters
    print(f"\n{'='*60}")
    print(f"Trial {trial.number} - Testing parameters:")
    print(f"  Fx={Fx:.2f}, Fy={Fy:.2f}, Fz={Fz:.2f}, f0={f0:.3f}, f1={f1:.2f}")
    print(f"{'='*60}")

    # Reset the simulation
    simulator.reset()

    # Extract simulation options
    sim_options = simulator.options

    # Determine number of jumps to simulate
    # n_jumps = 5 # Feel free to change this number
    n_jumps = 3  # for lateral jump like quadruped_jump.py

    # Using average of f0 for jump duration calculation
    f0_avg = f0
    # jump_duration = 1/(2*f0_avg) + 1/(2*f1)
    jump_duration = 3 # for lateral jump like quadruped_jump.py
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
    airborne = 0

    # Jump detection state machine
    jump_state = "INITIAL"  # States: INITIAL, GROUNDED, AIRBORNE
    jump_distances = []  # Store distance for each successful jump
    jump_start_pos = None  # Position when jump starts (recorded at takeoff)
    jump_start_yaw = None  # Yaw angle when jump starts (for twist objective)
    failed = False  # Whether the robot has failed (tipped over)
    first_grounded = False  # Track if we've seen the first stable grounded state
    jump_count = 0  # Count total jumps to skip the first one

    for step_idx in range(n_steps):
        # Step the oscillator
        for profile in force_profiles:
            profile.step(sim_options.timestep)

        # Determine airborne status
        foot_contacts = simulator.get_foot_contacts()
        if all(foot_contacts):
            airborne = 0
        if not any(foot_contacts):
            airborne = 1

        # Compute torques as motor targets (reuses your controller functions)
        tau = np.zeros(N_JOINTS * N_LEGS)
        if airborne == 1:
            tau += nominal_position_fly(simulator, Kp, Kd, Kd_point)
        elif airborne == 0:
            tau += nominal_position_ground(simulator, Kp, Kd, Kd_point)
        tau += apply_force_profile(simulator, force_profiles)
        tau += gravity_compensation(simulator)

        # If touching the ground, add virtual model
        on_ground = any(foot_contacts)
        if on_ground:
            tau += virtual_model(simulator, Katt)

        # Set the motor commands and step the simulation
        simulator.set_motor_targets(tau)
        simulator.step()

        # Get current state
        current_pos = simulator.get_base_position()
        current_rpy = simulator.get_base_orientation_roll_pitch_yaw()
        base_position_history.append(current_pos)
        contact_history.append(foot_contacts)
        base_row_pitch_yaw_history.append(current_rpy)

        # Check for failure (tipped over)
        roll, pitch = current_rpy[0], current_rpy[1]
        roll_modify = 8/9
        if abs(roll) >= np.pi/2 * roll_modify or abs(pitch) >= np.pi/2:
            failed = True
            break

        # Jump state machine
        all_feet_grounded = all(foot_contacts)
        all_feet_airborne = not any(foot_contacts)

        if jump_state == "INITIAL":
            # Wait for first stable grounded state
            if all_feet_grounded:
                first_grounded = True
                jump_start_pos = current_pos.copy()
                jump_start_yaw = current_rpy[2]
                jump_state = "GROUNDED"

        elif jump_state == "GROUNDED":
            if all_feet_airborne:
                # Jump started - we already have start position from grounded state
                jump_state = "AIRBORNE"

        elif jump_state == "AIRBORNE":
            if all_feet_grounded:
                # Jump completed - calculate distance
                jump_end_pos = current_pos.copy()
                jump_end_yaw = current_rpy[2]

                if objective_choice == "distance":
                    jump_dist = jump_end_pos[0] - jump_start_pos[0]
                elif objective_choice == "lateral_distance":
                    jump_dist = abs(jump_end_pos[1] - jump_start_pos[1])
                elif objective_choice == "twist":
                    jump_dist = abs(jump_end_yaw - jump_start_yaw)

                jump_count += 1

                # Skip the first jump (usually a tiny initial movement ~0.0004m)
                if jump_count > 1:
                    jump_distances.append(jump_dist)
                    print(f"  Jump {len(jump_distances)} completed: distance = {jump_dist:.4f}")
                else:
                    print(f"  Jump {jump_count} (initial, skipped): distance = {jump_dist:.4f}")

                # Update start position for next jump
                jump_start_pos = current_pos.copy()
                jump_start_yaw = current_rpy[2]
                jump_state = "GROUNDED"

                # Stop after 5 successful jumps (not counting the first one)
                if len(jump_distances) >= 5:
                    break

    # Calculate objective value based on average jump distance
    print(f"\n{'-'*60}")
    if failed:
        objective_value = 0.0
        print(f"Trial {trial.number} FAILED: Robot tipped over")
    elif len(jump_distances) == 0:
        objective_value = 0.0
        print(f"Trial {trial.number}: No successful jumps completed")
    else:
        # Take up to first 5 jumps and calculate average
        valid_jumps = jump_distances[:min(5, len(jump_distances))]
        objective_value = sum(valid_jumps) / len(valid_jumps)

        print(f"Trial {trial.number} Summary:")
        print(f"  Total jumps completed: {len(valid_jumps)}")
        print(f"  Jump distances: {[f'{d:.4f}' for d in valid_jumps]}")

        if objective_choice == "distance":
            print(f"  Average distance: {objective_value:.4f} m")
        elif objective_choice == "lateral_distance":
            print(f"  Average lateral distance: {objective_value:.4f} m")
        elif objective_choice == "twist":
            print(f"  Average twist: {objective_value:.4f} rad")

    print(f"{'='*60}\n")
    return objective_value


def replay_jump(simulator: QuadSimulator, params: dict, objective_choice: str):
    """
    Simulates a jump with the given parameters.
    This function is used to replay the best jump found during optimization.
    """
    # Extract parameters based on objective
    if objective_choice == "distance":
        Fx = params["Fx"]
        Fz = params["Fz"]
        f0 = params["f0"]
        Fy = 0.0
    elif objective_choice == "lateral_distance":
        Fy = params["Fy"]
        Fz = params["Fz"]
        f0 = params["f0"]
        Fx = -18.0  # Must match the value in evaluate_jumping
    elif objective_choice == "twist":
        Fx = params["Fx"]
        Fy = params["Fy"]
        f0 = params["f0"]
        Fz = 250.0  # Must match the value in evaluate_jumping
    else:
        raise ValueError(f"Invalid objective choice for replay: {objective_choice}")

    f1 = 0.8  # Must match the value in evaluate_jumping

    # Reset the simulation
    simulator.reset()

    # Extract simulation options
    sim_options = simulator.options

    # Determine number of jumps to simulate
    # n_jumps = 5 # Simulate for long enough to capture one full jump
    n_jumps = 3  # for lateral jump like quadruped_jump.py

    # Using average of f0 for jump duration calculation
    # f0_avg = f0
    # jump_duration = 1/(2*f0_avg) + 1/(2*f1)
    jump_duration = 3 # for lateral jump like quadruped_jump.py
    n_steps = int(n_jumps * jump_duration / sim_options.timestep)

    # Create separate profiles for each leg
    profile_FR = FootForceProfile(f0=f0, f1=f1, Fx=Fx, Fy=Fy, Fz=Fz)
    profile_FL = FootForceProfile(f0=f0, f1=f1, Fx=Fx, Fy=Fy, Fz=Fz)
    profile_RR = FootForceProfile(f0=f0, f1=f1, Fx=Fx, Fy=Fy, Fz=Fz)
    profile_RL = FootForceProfile(f0=f0, f1=f1, Fx=Fx, Fy=Fy, Fz=Fz)

    # The order of legs is FR, FL, RR, RL
    force_profiles = [profile_FR, profile_FL, profile_RR, profile_RL]

    airborne = 0
    for _ in range(n_steps):
        # Step the oscillator
        for profile in force_profiles:
            profile.step(sim_options.timestep)

        # Determine airborne status
        foot_contacts = simulator.get_foot_contacts()
        if all(foot_contacts):
            airborne = 0
        if not any(foot_contacts):
            airborne = 1

        # Compute torques as motor targets (reuses your controller functions)
        tau = np.zeros(N_JOINTS * N_LEGS)
        if airborne == 1:
            tau += nominal_position_fly(simulator, Kp, Kd, Kd_point)
        elif airborne == 0:
            tau += nominal_position_ground(simulator, Kp, Kd, Kd_point)
        tau += apply_force_profile(simulator, force_profiles)
        tau += gravity_compensation(simulator)

        # If touching the ground, add virtual model
        on_ground = any(foot_contacts)
        if on_ground:
            tau += virtual_model(simulator, Katt)

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
