import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..bus import Bus


class PiperFollower:
    def __init__(self, channel: str, *, dt: float, timeout: float = 1.0):
        self.dt = dt
        self.timeout: float = timeout

        self.bus: Bus = Bus(channel, version="default")

        self.bus.connect()

        self.bus.enable_torque()

        self.bus.move_joints([0.0] * 6)
        self.bus.move_gripper(0.0)

    def move(self, qpos: ArrayLike) -> None:
        qpos = np.asarray(qpos, dtype=np.float64)

        joints = qpos[:6]
        gripper = qpos[6]

        self.bus.move_joints(joints)
        self.bus.move_gripper(gripper)

    def move_to_zero(self) -> None:
        self.bus.move_joints([0.0] * 6)
        self.bus.move_gripper(0.0)

    def get_state(self) -> dict[str, NDArray[np.float64]]:
        joints = self.bus.read_joints()
        gripper = self.bus.read_gripper()

        qpos = np.append(joints.position, gripper.width)
        qvel = np.append(joints.velocity, gripper.force)

        return {"qpos": qpos, "qvel": qvel}


class PiperTrajectoryFollower:
    def __init__(self, channel: str, *, timeout: float = 1.0):
        self.timeout: float = timeout

        self.bus: Bus = Bus(channel, version="default")

        self.bus.connect()

        self.bus.enable_torque()

        self.bus.move_joints([0.0] * 6)
        self.bus.move_gripper(0.0)

    def move(self, qpos: ArrayLike) -> None:
        qpos = np.asarray(qpos, dtype=np.float64)

        joints = qpos[:6]
        gripper = qpos[6]

        self.bus.move_joints(joints)
        self.bus.move_gripper(gripper)

    def move_to_zero(self) -> None:
        self.bus.move_joints([0.0] * 6)
        self.bus.move_gripper(0.0)

    def get_state(self) -> dict[str, NDArray[np.float64]]:
        joints = self.bus.read_joints()
        gripper = self.bus.read_gripper()

        qpos = np.append(joints.position, gripper.width)
        qvel = np.append(joints.velocity, gripper.force)

        return {"qpos": qpos, "qvel": qvel}
