import time
import pybullet
import numpy as np
from typing import Literal, Optional
from dataclasses import dataclass

from env import utils

PATH_URDF_PLANE = utils.get_pybullet_filepath("plane.urdf")


@dataclass
class SimulationOptions:
    # Whether to have an interactive GUI or not
    render: bool = True
    # Whether to record a video of the simulation to file, needs render=True
    record_video: bool = False
    # Simulation timestep in seconds
    timestep: float = 1e-3
    # Whether to attach the robot in air, helpful for debugging
    on_rack: bool = False
    # Whether the camera follows the robot in the GUI
    tracking_camera: bool = True
    # Control mode of the motors
    control_mode: Literal["POSITION", "TORQUE"] = "TORQUE"


class QuadSimulator:
    """Simulation of a Unitree A1 quadruped"""

    def __init__(
        self,
        options: SimulationOptions,
        config: str = "a1_description/config.yaml",
    ):
        """
        Initialize a pybullet simulation for a quadruped.

        Args:
            options (SimulationOptions): Options for the simulation.
            config (str, optional): Relative project filename of YAML config file for the quadruped to simulate. Defaults to "a1_description/config.yaml".
        """
        self.options = options
        self.config = utils.load_yaml(utils.get_project_filepath(config))

        # Create pybullet client
        if options.render:
            self._pybullet_client = pybullet.connect(
                method=pybullet.GUI, options="--nogui"
            )
        else:
            self._pybullet_client = pybullet.connect(method=pybullet.DIRECT)

        # Record simulation time
        self._time_s = 0.0

        # Recording video ID
        self._video_log_id: Optional[int] = None

        # Load model into pybullet
        self.reload_model()

    # --- COMMON SIMULATOR FUNCTIONS ---
    def reload_model(self):
        """Load the quadruped and world model into simulation"""
        # Disable rendering when setting up models (otherwise too slow)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        # Clear the simulation
        pybullet.resetSimulation(self._pybullet_client)

        # Load URDFs
        urdf_path = utils.get_project_filepath(self.config["urdf"])
        self._plane = pybullet.loadURDF(PATH_URDF_PLANE, basePosition=(0, 0, 0))
        self._body = pybullet.loadURDF(
            urdf_path,
            basePosition=self.config["init_position"],
            flags=pybullet.URDF_USE_SELF_COLLISION,
        )

        # Point camera towards model
        pybullet.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=45,
            cameraPitch=-10,
            cameraTargetPosition=self.config["init_position"],
        )

        # Set simulation options
        pybullet.setGravity(0, 0, -9.81)
        pybullet.setTimeStep(self.options.timestep)

        # Add constraint if on rack
        if self.options.on_rack:
            init_position = np.add(self.config["init_position"], [0, 0, 0.1])
            pybullet.createConstraint(
                self._body,
                -1,
                -1,
                -1,
                pybullet.JOINT_FIXED,
                [0, 0, 0],
                [0, 0, 0],
                init_position,
            )

        # Enable rendering again
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

        # Update internal model references
        self._build_motor_ids_list()
        self._build_defaults()
        self._record_masses()
        self._build_hip_offsets()

        # Internal reset
        self.reset()

        # Start the video recording
        if self.options.record_video:
            self._start_video_recording()

    def reset(self):
        """Reset the simulation to the initial timestep"""
        for motor_id, angle in zip(self._motor_ids, self._default_motor_angles):
            # Set default joint angles
            pybullet.resetJointState(
                self._body,
                jointIndex=motor_id,
                targetValue=angle,
                targetVelocity=0.0,
            )
            # Remove default pybullet joint damping
            pybullet.changeDynamics(
                self._body,
                motor_id,
                linearDamping=0,
                angularDamping=0,
            )

        # Reset the position (if not on rack)
        if not self.options.on_rack:
            pybullet.resetBasePositionAndOrientation(
                self._body, self.config["init_position"], [0, 0, 0, 1]
            )
            pybullet.resetBaseVelocity(self._body, [0, 0, 0], [0, 0, 0])

        # If using torque control, remove default velocity motors
        if self.options.control_mode == "TORQUE":
            pybullet.setJointMotorControlArray(
                self._body,
                self._motor_ids,
                controlMode=pybullet.VELOCITY_CONTROL,
                forces=[0] * len(self._motor_ids),
            )

        # Reset time
        self._time_s = 0
        self._start_time = 0

    def step(self):
        """Advance the simulation by one timestep"""
        # Log start time
        if self._time_s == 0:
            self._start_time = time.monotonic()

        pybullet.stepSimulation(self._pybullet_client)
        self._time_s += self.options.timestep

        if self.options.render:
            if self.options.tracking_camera:
                # Point camera towards model
                pybullet.resetDebugVisualizerCamera(
                    cameraDistance=1.0,
                    cameraYaw=45,
                    cameraPitch=-10,
                    cameraTargetPosition=self.get_base_position(),
                )
            target_time = self._start_time + self._time_s
            time_diff = max(0, target_time - time.monotonic())
            time.sleep(time_diff)

    def is_connected(self) -> bool:
        """Whether the simulation client is still open"""
        return pybullet.isConnected(self._pybullet_client)

    def close(self):
        """Close the simulation client"""
        if self.options.record_video:
            self._stop_video_recording()
        pybullet.disconnect(self._pybullet_client)

    def time(self) -> float:
        """Get simulation time in seconds"""
        return self._time_s

    # --- COMMON CONTROL FUNCTIONS ---
    def set_motor_targets(
        self, targets: list[float], motor_ids: Optional[list[int]] = None
    ):
        """Set motor targets (torques or angles depending on control mode)"""
        motor_ids = motor_ids or self._motor_ids
        if self.options.control_mode == "POSITION":
            pybullet.setJointMotorControlArray(
                self._body,
                jointIndices=motor_ids,
                controlMode=pybullet.POSITION_CONTROL,
                targetPositions=targets,
                forces=[self.config["limit_torque"]] * len(motor_ids),
            )
        elif self.options.control_mode == "TORQUE":
            pybullet.setJointMotorControlArray(
                self._body,
                jointIndices=motor_ids,
                controlMode=pybullet.TORQUE_CONTROL,
                forces=targets,
            )

    # --- COMMON MODEL FUNCTIONS ---
    def get_mass(self) -> float:
        """Get the total robot mass computed from the URDF"""
        return self._total_mass

    def get_joint_names(self):
        """Get the names of all the joints of the quadruped"""
        return [
            pybullet.getJointInfo(self._body, i)[1].decode("UTF-8")
            for i in range(self.get_num_joints_total())
        ]

    def get_num_joints_total(self):
        """Get the total number of joints of the quadruped"""
        return pybullet.getNumJoints(self._body)

    def get_motor_angles(self, leg_id: Optional[int] = None) -> np.ndarray:
        """Get current motor angles"""
        q = np.array(
            [state[0] for state in pybullet.getJointStates(self._body, self._motor_ids)]
        )
        if leg_id is None:
            return q
        n_joints = self.config["n_motors_per_leg"]
        return q[leg_id * n_joints : leg_id * n_joints + n_joints]

    def get_motor_velocities(self, leg_id: Optional[int] = None) -> np.ndarray:
        """Get current motor velocities"""
        dq = np.array(
            [state[1] for state in pybullet.getJointStates(self._body, self._motor_ids)]
        )
        if leg_id is None:
            return dq
        n_joints = self.config["n_motors_per_leg"]
        return dq[leg_id * n_joints : leg_id * n_joints + n_joints]

    def get_motor_torques(self, leg_id: Optional[int] = None) -> np.ndarray:
        """Get current applied motor torques"""
        tau = np.array(
            [state[3] for state in pybullet.getJointStates(self._body, self._motor_ids)]
        )
        if leg_id is None:
            return tau
        n_joints = self.config["n_motors_per_leg"]
        return tau[leg_id * n_joints : leg_id * n_joints + n_joints]

    def get_base_position(self) -> np.ndarray:
        """Get 3D position of the robot base in world frame."""
        pos, _ = pybullet.getBasePositionAndOrientation(self._body)
        return np.array(pos)

    def get_base_orientation_quaternion(self) -> np.ndarray:
        """Get quaternion of the robot base in world frame."""
        _, orn = pybullet.getBasePositionAndOrientation(self._body)
        return np.array(orn)

    def get_base_orientation_matrix(self) -> np.ndarray:
        """Get orientation matrix of the robot base with respect to the world frame."""
        quat = self.get_base_orientation_quaternion()
        return np.array(pybullet.getMatrixFromQuaternion(quat)).reshape((3, 3))

    def get_base_orientation_roll_pitch_yaw(self) -> np.ndarray:
        """Get euler angles of the robot base in world frame."""
        quat = self.get_base_orientation_quaternion()
        return np.array(pybullet.getEulerFromQuaternion(quat))

    def get_base_linear_velocity(self) -> np.ndarray:
        """Get 3D linear velocity of the robot base in world frame."""
        vel, _ = pybullet.getBaseVelocity(self._body)
        return np.array(vel)

    def get_base_angular_velocity(self) -> np.ndarray:
        """Get angular velocity of the robot base in world frame."""
        _, ang = pybullet.getBaseVelocity(self._body)
        return np.array(ang)

    def get_jacobian_and_position(self, leg_id: int) -> np.ndarray:
        """Get Jacobian foot position of leg with given id in leg frame:
        0: FR; 1: FL; 2: RR; 3: RL;
        """
        # Retrieve link lengths
        l1, l2, l3 = (
            self.config["link_length_hip"],
            self.config["link_length_thigh"],
            self.config["link_length_calf"],
        )

        # Retrieve joint angles
        q1, q2, q3 = self.get_motor_angles(leg_id)

        # Right side hip angle is inverted
        side_sign = -1 if leg_id in (0, 2) else 1

        # Compute sines/cosines
        s1 = np.sin(q1)
        s2 = np.sin(q2)
        c1 = np.cos(q1)
        c2 = np.cos(q2)
        s23 = np.sin(q2 + q3)
        c23 = np.cos(q2 + q3)

        # Compute Jacobian
        J = np.zeros((3, 3))
        J[0, 0] = 0
        J[1, 0] = -side_sign * l1 * s1 + c1 * (l2 * c2 + l3 * c23)
        J[2, 0] = side_sign * l1 * c1 + s1 * (l2 * c2 + l3 * c23)
        J[0, 1] = -l2 * c2 - l3 * c23
        J[1, 1] = -l2 * s1 * s2 - l3 * s1 * s23
        J[2, 1] = l2 * c1 * s2 + l3 * c1 * s23
        J[0, 2] = -l3 * c23
        J[1, 2] = -l3 * s1 * s23
        J[2, 2] = l3 * c1 * s23

        # Compute foot position
        x = -l2 * s2 - l3 * s23
        y = side_sign * l1 * c1 + s1 * (l2 * c2 + l3 * c23)
        z = side_sign * l1 * s1 - c1 * (l2 * c2 + l3 * c23)

        return J, np.array((x, y, z))

    def get_world_foot_position(self, leg_id: int) -> np.ndarray:
        """Get foot position of leg with given id in world frame:
        0: FR; 1: FL; 2: RR; 3: RL;
        """
        com_pos = self.get_base_position()
        _, foot_pos = self.get_jacobian_and_position(leg_id)
        hip_offset = self._hip_offsets[leg_id]
        R = self.get_base_orientation_matrix()

        # FIXME: side sign is specific to A1
        hip_world = com_pos + R @ hip_offset
        foot_world = hip_world + R @ foot_pos
        return foot_world

    def get_foot_contacts(self) -> np.ndarray:
        """Get a boolean array to check whether each foot is touching the ground.
        Indexed by leg id:
        0: FR; 1: FL; 2: RR; 3: RL;
        """
        contact_bools = [False] * len(self._foot_link_ids)
        for i, foot_link in enumerate(self._foot_link_ids):
            contacts = pybullet.getContactPoints(
                bodyA=-1, bodyB=self._body, linkIndexB=foot_link
            )
            if contacts is not None and len(contacts) > 0:
                contact_bools[i] = True
        return np.array(contact_bools, dtype=bool)

    # --- INTERNAL FUNCTIONS ---
    def _build_motor_ids_list(self):
        joint_names = self.get_joint_names()
        self._hip_joints = [
            i
            for i in range(self.get_num_joints_total())
            if utils.match_joint_type(joint_names[i], "hip")
        ]
        self._thigh_joints = [
            i
            for i in range(self.get_num_joints_total())
            if utils.match_joint_type(joint_names[i], "thigh")
        ]
        self._calf_joints = [
            i
            for i in range(self.get_num_joints_total())
            if utils.match_joint_type(joint_names[i], "calf")
        ]
        self._foot_link_ids = [
            i
            for i in range(self.get_num_joints_total())
            if utils.match_joint_type(joint_names[i], "foot")
        ]

        self._motor_ids = []
        for hip, thigh, calf in zip(
            self._hip_joints, self._thigh_joints, self._calf_joints
        ):
            self._motor_ids.extend([hip, thigh, calf])

    def _build_defaults(self):
        self._default_motor_angles = np.array(
            [
                self.config["default_angles"]["hip"],
                self.config["default_angles"]["thigh"],
                self.config["default_angles"]["calf"],
            ]
            * self.config["n_legs"]
        )

    def _record_masses(self):
        self._total_mass = np.sum(
            [
                pybullet.getDynamicsInfo(self._body, i)[0]
                for i in range(-1, self.get_num_joints_total())  # Include -1 for base
            ]
        )

    def _build_hip_offsets(self):
        # FIXME: this is specific to the A1 and might break for other embodiments, extract from the URDF in the future
        offset_x = self.config["com_hip_offset_x"]
        offset_y = self.config["com_hip_offset_y"]

        offsets = []
        # FR
        offsets.append([offset_x, -offset_y, 0])
        # FL
        offsets.append([offset_x, offset_y, 0])
        # RR
        offsets.append([-offset_x, -offset_y, 0])
        # RL
        offsets.append([-offset_x, offset_y, 0])

        self._hip_offsets = np.asarray(offsets)

    def _start_video_recording(self):
        # If recording is ongoing, stop it before starting a new one
        if self._video_log_id != None:
            self._stop_video_recording()

        filename = utils.generate_video_filename()
        self._video_log_id = pybullet.startStateLogging(
            pybullet.STATE_LOGGING_VIDEO_MP4, filename
        )

    def _stop_video_recording(self):
        pybullet.stopStateLogging(self._video_log_id)
        self._video_log_id = None
