from dataclasses import dataclass
from typing import Optional

import numpy as np
import placo


@dataclass
class PiperIK:
    robot: placo.RobotWrapper
    solver: placo.KinematicsSolver

    effector_task: placo.FrameTask
    gripper_task: placo.JointsTask

    effector_name: str
    gripper_name: str = "gripper"

    def __init__(self, urdf_path: str, effector_name: str = "flange_joint"):
        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)

        self.effector_name = effector_name

        gear_task = self.solver.add_gear_task()
        gear_task.configure("gear", "hard")

        gear_task.set_gear("gripper_joint1", "gripper", 0.5)
        gear_task.set_gear("gripper_joint2", "gripper", -0.5)

        self.effector_task = self.solver.add_frame_task(self.effector_name, np.eye(4))

        self.gripper_task = self.solver.add_joints_task()

    def set_joints(self, joints, gripper: Optional[float] = None):
        for i, joint in enumerate(joints):
            self.robot.set_joint(f"joint{i + 1}", joint)

        if gripper is not None:
            assert 0.1 >= gripper >= 0
            self.robot.set_joint("gripper", gripper)
            self.robot.set_joint("gripper_joint1", gripper / 2)
            self.robot.set_joint("gripper_joint2", -gripper / 2)

        self.robot.update_kinematics()

    def get_joints(self):
        joints = [self.robot.get_joint(f"joint{i + 1}") for i in range(6)]
        gripper = self.robot.get_joint("gripper")
        return joints, gripper

    def get_frame(self, name: Optional[str] = None):
        if name is None:
            name = self.effector_name

        return self.robot.get_T_world_frame(name)

    def solve(self, dt: Optional[float] = None):
        if dt is not None:
            self.solver.dt = dt

        self.solver.solve(True)
        self.robot.update_kinematics()

    @property
    def dt(self):
        return self.solver.dt

    @dt.setter
    def dt(self, value: float):
        self.solver.dt = value

    def set_goal(self, frame, gripper):
        self.effector_task.T_world_frame = frame
        self.gripper_task.set_joint(self.gripper_name, gripper)

    def get_goal_frame(self):
        frame = self.effector_task.T_world_frame

        return frame

    @property
    def q(self):
        return self.robot.state.q
