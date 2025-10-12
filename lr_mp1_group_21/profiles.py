import numpy as np


class FootForceProfile:
    """Class to generate foot force profiles over time using a single CPG oscillator"""

    def __init__(self, f0: float, f1: float, Fx: float, Fy: float, Fz: float):
        """
        Create instance of foot force profile with its arguments.

        Args:
            f0 (float): Frequency of the impulse (Hz)
            f1 (float): Frequency between impulses (Hz)
            Fx (float): Foot force amplitude in X direction (N)
            Fy (float): Foot force amplitude in Y direction (N)
            Fz (float): Foot force amplitude in Z direction (N)
        """
        self.theta = 0
        self.f0 = f0
        self.f1 = f1
        self.F = np.array([Fx, Fy, Fz])

    def step(self, dt: float):
        """
        Step the oscillator by a single timestep.

        Args:
            dt (float): Timestep duration (s)
        """
        # TODO: integrate the oscillator equation
        freq = self.f1 if np.sin(self.theta) >= 0 else self.f0  # apply the impulse or between impulse frequency
        # if from [0, pi], then f1, so sin>0
        self.theta = (self.theta + 2.0 * np.pi * freq * dt) % (2.0 * np.pi)

    def phase(self) -> float:
        """Get oscillator phase in [0, 2pi] range."""
        # TODO: return the phase of the oscillator in [0, 2pi] range
        return self.theta  # theta is the phase?

    def force(self) -> np.ndarray:
        """
        Get force vector of the force profile at the current timestep.

        Returns:
            np.ndarray: An R^3 array [Fx, Fy, Fz]
        """
        # TODO: return the force vector given the oscillator state
        sintheta = np.sin(self.theta)
        if sintheta<0:
            return (self.F).T*sintheta
        return np.zeros(3)

    def impulse_duration(self) -> float:
        """Return impulse duration in seconds."""
        # TODO: compute the impulse duration in seconds
        return 0.5/self.f0

    def idle_duration(self) -> float:
        """Return idle time between impulses in seconds"""
        # TODO: compute the idle duration in seconds
        return 0.5/self.f1
