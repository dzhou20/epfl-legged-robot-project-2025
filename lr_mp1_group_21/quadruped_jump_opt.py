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
        "Fz_FR": 150.0, "Fx_FR": 75.0, "Fy_FR": 0.0, "f0_FR": 1.25,
        "Fz_FL": 150.0, "Fx_FL": 75.0, "Fy_FL": 0.0, "f0_FL": 1.25,
        "Fz_RR": 300.0, "Fx_RR": 75.0, "Fy_RR": 0.0, "f0_RR": 1.25,
        "Fz_RL": 300.0, "Fx_RL": 75.0, "Fy_RL": 0.0, "f0_RL": 1.25,
    }
    study.enqueue_trial(initial_params)

    # Run the optimization
    # You can change the number of trials here (the maximum number of trials is 50)
    study.optimize(objective, n_trials=20)

    # Close the simulation
    simulator.close()

    # Log the results
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
    last_trial = study.trials[-1]
    print("Last trial value:", last_trial.value)
    print("Last trial params:", last_trial.params)

    # Replay the best jump with rendering
    print("\nReplaying the best jump...")
    replay_sim_options = SimulationOptions(render=True, on_rack=False, tracking_camera=True)
    replay_simulator = QuadSimulator(replay_sim_options)
    replay_jump(replay_simulator, study.best_params)

    # Replay the last jump with rendering
    print("\nReplaying the last jump...")
    replay_jump(replay_simulator, last_trial.params)
    replay_simulator.close()


    # OPTIONAL: add additional functions here (e.g., plotting, recording to file)
    # E.g., cycling through all the evaluated parameters and values:
    for trial in study.get_trials():
        trial.number  # Number of the trial
        trial.params  # Used parameters
        trial.value  # Resulting objective function value


def evaluate_jumping(trial: Trial, simulator: QuadSimulator, objective_choice: str) -> float:
    # the number of four legs: FR(Front Right), FL(Front Left), RR(Rear Right), RL(Rear Left)
    # Parameters for each leg
    Fz_FR = trial.suggest_float("Fz_FR", 150, 350)
    Fx_FR = trial.suggest_float("Fx_FR", 0, 150)
    Fy_FR = trial.suggest_float("Fy_FR", 0, 150)
    f0_FR = trial.suggest_float("f0_FR", 0.75, 1.75)

    Fz_FL = trial.suggest_float("Fz_FL", 150, 350)
    Fx_FL = trial.suggest_float("Fx_FL", 0, 150)
    Fy_FL = trial.suggest_float("Fy_FL", 0, 150)
    f0_FL = trial.suggest_float("f0_FL", 0.75, 1.75)

    Fz_RR = trial.suggest_float("Fz_RR", 150, 350)
    Fx_RR = trial.suggest_float("Fx_RR", 0, 150)
    Fy_RR = trial.suggest_float("Fy_RR", 0, 150)
    f0_RR = trial.suggest_float("f0_RR", 0.75, 1.75)

    Fz_RL = trial.suggest_float("Fz_RL", 150, 350)
    Fx_RL = trial.suggest_float("Fx_RL", 0, 150)
    Fy_RL = trial.suggest_float("Fy_RL", 0, 150)
    f0_RL = trial.suggest_float("f0_RL", 0.75, 1.75)
    
    f1 = 1.25 # This seems to be a constant

    # Reset the simulation
    simulator.reset()

    # Extract simulation options
    sim_options = simulator.options

    # Determine number of jumps to simulate
    if objective_choice == "first_jump_distance":
        n_jumps = 2  # Simulate for long enough to capture one full jump
    else:
        n_jumps = 10  # Feel free to change this number
    
    # Using average of f0 for jump duration calculation
    f0_avg = (f0_FR + f0_FL + f0_RR + f0_RL) / 4.0
    jump_duration = 1/(2*f0_avg) + 1/(2*f1)
    n_steps = int(n_jumps * jump_duration / sim_options.timestep)

    # Create separate profiles for each leg
    profile_FR = FootForceProfile(f0=f0_FR, f1=f1, Fx=Fx_FR, Fy=Fy_FR, Fz=Fz_FR)
    profile_FL = FootForceProfile(f0=f0_FL, f1=f1, Fx=Fx_FL, Fy=Fy_FL, Fz=Fz_FL)
    profile_RR = FootForceProfile(f0=f0_RR, f1=f1, Fx=Fx_RR, Fy=Fy_RR, Fz=Fz_RR)
    profile_RL = FootForceProfile(f0=f0_RL, f1=f1, Fx=Fx_RL, Fy=Fy_RL, Fz=Fz_RL)

    # The order of legs is FR, FL, RR, RL
    force_profiles = [profile_FR, profile_FL, profile_RR, profile_RL]
    
    base_position_history = []
    contact_history = []
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

    base_position_history = np.array(base_position_history)

    def get_max_height(history: np.ndarray) -> float:
        """Get the maximum height reached by the base during the simulation."""
        max_height = np.max(history[:, 2])  # Z coordinate is the height
        return max_height

    def get_max_distance(history: np.ndarray) -> float:
        """Get the maximum distance covered by the base during the simulation."""
        final_pos = history[-1]
        max_dist = np.sqrt(final_pos[0]**2 + final_pos[1]**2)
        return max_dist

    def get_first_jump_distance(history: np.ndarray, contacts_hist: list) -> float:
        """
        Calculates the distance of the first successful jump.
        A jump is successful if the robot leaves the ground and lands again.
        """
        liftoff_idx = -1
        landing_idx = -1

        # Find liftoff (all feet off ground)
        for i, contacts in enumerate(contacts_hist):
            if not np.any(contacts):
                liftoff_idx = i
                break
        
        if liftoff_idx == -1:
            return 0.0 # No liftoff detected

        # Find landing (at least one foot on ground after liftoff)
        for i in range(liftoff_idx, len(contacts_hist)):
            if np.any(contacts_hist[i]):
                landing_idx = i
                break
        
        if landing_idx == -1:
            return 0.0 # No landing detected after liftoff

        liftoff_pos = history[liftoff_idx]
        landing_pos = history[landing_idx]

        distance = np.sqrt((landing_pos[0] - liftoff_pos[0])**2 + (landing_pos[1] - liftoff_pos[1])**2)
        return distance

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
    Fz_FR = params["Fz_FR"]
    Fx_FR = params["Fx_FR"]
    Fy_FR = params["Fy_FR"]
    f0_FR = params["f0_FR"]

    Fz_FL = params["Fz_FL"]
    Fx_FL = params["Fx_FL"]
    Fy_FL = params["Fy_FL"]
    f0_FL = params["f0_FL"]

    Fz_RR = params["Fz_RR"]
    Fx_RR = params["Fx_RR"]
    Fy_RR = params["Fy_RR"]
    f0_RR = params["f0_RR"]

    Fz_RL = params["Fz_RL"]
    Fx_RL = params["Fx_RL"]
    Fy_RL = params["Fy_RL"]
    f0_RL = params["f0_RL"]
    
    f1 = 1.25 # This seems to be a constant

    # Reset the simulation
    simulator.reset()

    # Extract simulation options
    sim_options = simulator.options

    # Determine number of jumps to simulate
    n_jumps = 2 # Simulate for long enough to capture one full jump
    
    # Using average of f0 for jump duration calculation
    f0_avg = (f0_FR + f0_FL + f0_RR + f0_RL) / 4.0
    jump_duration = 1/(2*f0_avg) + 1/(2*f1)
    n_steps = int(n_jumps * jump_duration / sim_options.timestep)

    # Create separate profiles for each leg
    profile_FR = FootForceProfile(f0=f0_FR, f1=f1, Fx=Fx_FR, Fy=Fy_FR, Fz=Fz_FR)
    profile_FL = FootForceProfile(f0=f0_FL, f1=f1, Fx=Fx_FL, Fy=Fy_FL, Fz=Fz_FL)
    profile_RR = FootForceProfile(f0=f0_RR, f1=f1, Fx=Fx_RR, Fy=Fy_RR, Fz=Fz_RR)
    profile_RL = FootForceProfile(f0=f0_RL, f1=f1, Fx=Fx_RL, Fy=Fy_RL, Fz=Fz_RL)

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
