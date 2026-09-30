import numpy as np
import placo
from numpy.typing import ArrayLike, NDArray


class Dynamics:
    robot: placo.RobotWrapper
    solver: placo.DynamicsSolver

    effector_task: placo.DynamicsFrameTask

    def __init__(
        self,
        urdf_path: str,
        *,
        effector_name: str,
        gripper_name: str,
        dt: float = 0.001,
        pos_weight: float = 1.0,
        rot_weight: float = 0.1,
    ) -> None:
        super().__init__()

        self.effector_name = effector_name
        self.gripper_name = gripper_name

        self.robot = robot = placo.RobotWrapper(urdf_path)
        self.solver = solver = placo.DynamicsSolver(robot)

        self.joint_names = list(self.robot.joint_names())

        self.dt = dt

        solver.dt = dt

        solver.enable_joint_limits(True)
        solver.enable_velocity_limits(True)

        solver.mask_fbase(True)

        self.effector_task = effector_task = solver.add_frame_task(
            effector_name, np.eye(4)
        )
        effector_task.configure(effector_name, "soft", pos_weight, rot_weight)

        effector_task.T_world_frame = solver.robot.get_T_world_frame(effector_name)

        posture = solver.add_joints_task()
        posture.set_joints({name: 0.0 for name in self.joint_names})
        posture.configure("posture", "soft", 1e-4)

    def set_joints(self, joints: ArrayLike) -> None:
        joints = np.asarray(joints, dtype=np.float64)

        for name, joint in zip(self.joint_names[: len(joints)], joints):
            self.robot.set_joint(name, float(joint))

        self.robot.update_kinematics()

    def get_joints(self) -> list[float]:
        return [self.robot.get_joint(name) for name in self.joint_names]

    def set_gripper(self, width: float) -> None:
        self.robot.set_joint(self.gripper_name, width)

        self.robot.update_kinematics()

    def get_gripper(self) -> float:
        return self.robot.get_joint("gripper")

    def forward(self) -> NDArray[np.float64]:
        frame = self.robot.get_T_world_frame(self.effector_name)
        return frame

    def inverse(
        self,
        frame: ArrayLike,
        *,
        vel: ArrayLike | None = None,
    ) -> list[float]:
        self.effector_task.T_world_frame = np.asarray(frame)

        if vel is not None:
            self.effector_task.position().dtarget_world = np.asarray(vel)

        self.solver.solve(True)
        self.robot.update_kinematics()

        joints = self.get_joints()
        return joints
