import numpy as np
import placo
from numpy.typing import ArrayLike


class EffectorPlanner:
    def __init__(self, dt, alpha: float = 1.0):
        self.dt = dt
        self.alpha = alpha

        self.start = np.eye(4)

    def set_start(self, frame: ArrayLike):
        self.start = np.asarray(frame, dtype=np.float64)

    def plan(self, end, dt):
        end = end * self.alpha + self.start * (1 - self.alpha)
        vel = (end[:3, -1] - self.start[:3, -1]) / dt

        times = np.linspace(0, 1, max(int(dt / self.dt), 2))
        times = times[1:]

        for t in times:
            waypoint = placo.interpolate_frames(self.start, end, t)

            yield waypoint, vel

        self.start = end


class JointPlanner:
    def __init__(self, dt, alpha: float = 1.0):
        self.dt = dt
        self.alpha = alpha

        self.start = np.zeros(7)

    def set_start(self, qpos: ArrayLike):
        self.start = np.asarray(qpos, dtype=np.float64)

    def plan(self, end, dt):
        end = end * self.alpha + self.start * (1 - self.alpha)
        vel = (end - self.start) / dt

        times = np.linspace(0, 1, max(int(dt / self.dt), 2))
        times = times[1:] * dt

        for t in times:
            waypoint = self.start + vel * t

            yield waypoint

        self.start = end
