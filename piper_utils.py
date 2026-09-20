from __future__ import annotations

import time
from dataclasses import dataclass
from functools import wraps

import numpy as np
import placo
from numpy.typing import ArrayLike
from pyAgxArm import (
    AgxArmFactory,
    ArmModel,
    create_agx_arm_config,
)
from pyAgxArm.protocols.can_protocol.msgs.core import MessageAbstract
from pyAgxArm.protocols.can_protocol.msgs.piper.default import ArmMsgFeedbackHighSpd


class Kinematics:
    robot: placo.RobotWrapper
    solver: placo.DynamicsSolver

    effector_task: placo.DynamicsFrameTask
    gripper_task: placo.DynamicsJointsTask

    def __init__(
        self,
        urdf_path: str,
        *,
        effector_name: str = "gripper_tcp",
        gripper_name: str = "gripper",
        dt: float = 0.008,
        pos_weight: float = 1.0,
        rot_weight: float = 5e-3,
        gripper_max: float = 0.1,
    ):
        self.effector_name: str = effector_name
        self.gripper_name: str = gripper_name
        self.dt: float = dt
        self.gripper_max: float = gripper_max

        self.robot = robot = placo.RobotWrapper(urdf_path)
        self.solver = solver = placo.DynamicsSolver(robot)

        solver.dt = dt

        solver.enable_joint_limits(True)
        solver.enable_velocity_limits(True)

        solver.mask_fbase(True)

        gears = solver.add_gear_task()
        gears.add_gear("gripper_joint1", "gripper", 0.5)
        gears.add_gear("gripper_joint2", "gripper", -0.5)
        gears.configure("gear", "hard")

        self.effector_task = effector_task = solver.add_frame_task(
            effector_name, np.eye(4)
        )
        effector_task.configure(effector_name, "soft", pos_weight, rot_weight)

        self.gripper_task = gripper_task = solver.add_joints_task()
        gripper_task.configure(gripper_name, "soft", 1.0)

        effector_task.T_world_frame = solver.robot.get_T_world_frame(effector_name)
        gripper_task.set_joint(gripper_name, 0)

        posture = solver.add_joints_task()
        posture.set_joints({f"joint{i + 1}": 0.0 for i in range(6)})
        posture.configure("posture", "soft", 1e-4)

    def set_qpos(self, joints: ArrayLike):
        joints = np.asarray(joints, dtype=np.float64)

        for i, joint in enumerate(joints[:6]):
            self.robot.set_joint(f"joint{i + 1}", joint)

        if len(joints) == 7:
            width = joints[6]

            self.robot.set_joint("gripper", width)
            self.robot.set_joint("gripper_joint1", width / 2)
            self.robot.set_joint("gripper_joint2", -width / 2)

        self.robot.update_kinematics()

    def get_qpos(self):
        joints = [self.robot.get_joint(f"joint{i + 1}") for i in range(6)]
        joints.append(self.robot.get_joint(self.gripper_name))
        return joints

    def forward(self):
        return self.robot.get_T_world_frame(self.effector_name)

    def inverse(
        self,
        frame: ArrayLike,
        gripper: float,
        *,
        vel: ArrayLike | None = None,
        ee_vel: float = 0,
    ):
        self.effector_task.T_world_frame = np.asarray(frame)

        if vel is not None:
            self.effector_task.position().dtarget_world = np.asarray(vel)

        self.gripper_task.set_joint(self.gripper_name, gripper, ee_vel)

        result = self.solver.solve(True)
        self.robot.update_kinematics()
        return result


def raise_timeout(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < self.timeout:
            out = fn(self, *args, **kwargs)

            if out:
                return out

        raise TimeoutError

    return wrapper


@dataclass(frozen=True)
class JointState:
    t: tuple[float, ...]
    hz: tuple[float, ...]
    pos: tuple[float, ...]
    vel: tuple[float, ...]


class RealPiper:
    def __init__(self, channel: str = "can0", timeout: float = 1.0):
        cfg = create_agx_arm_config(robot=ArmModel.PIPER, channel=channel)

        self.arm = robot = AgxArmFactory.create_arm(cfg)
        self.ee = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)

        robot.connect()

        if robot.has_comm_error():
            raise RuntimeError

        robot.set_speed_percent(100)
        robot.set_motion_mode("j")

        self.timeout: float = timeout

    def close(self):
        self.arm.disconnect()

    def enable_torque(self):
        t0 = time.perf_counter()

        while time.perf_counter() - t0 < self.timeout:
            if self.arm.enable():
                return

        raise TimeoutError

    def disable_torque(self):
        t0 = time.perf_counter()

        while time.perf_counter() - t0 < self.timeout:
            if self.arm.disable():
                return

        raise TimeoutError

    def send_qpos(self, qpos: ArrayLike, force: float = 1.0):
        qpos = np.asarray(qpos, dtype=np.float64)

        self.arm.move_j(list(qpos[:6]))
        self.ee.move_gripper_m(qpos[6], force)

    def recv_q(self) -> JointState:
        t0 = time.perf_counter()

        while time.perf_counter() - t0 < self.timeout:
            motors: list[MessageAbstract[ArmMsgFeedbackHighSpd]] = []

            for i in range(6):
                if i == 0:
                    ms = self.arm.get_motor_states(1)
                elif i == 1:
                    ms = self.arm.get_motor_states(2)
                elif i == 2:
                    ms = self.arm.get_motor_states(3)
                elif i == 3:
                    ms = self.arm.get_motor_states(4)
                elif i == 4:
                    ms = self.arm.get_motor_states(5)
                elif i == 5:
                    ms = self.arm.get_motor_states(6)
                else:
                    raise RuntimeError

                if ms is None:
                    break

                motors.append(ms)
            else:
                gs = self.ee.get_gripper_status()

                if gs is None:
                    continue

                t = [ms.timestamp for ms in motors]
                hz = [ms.hz for ms in motors]

                qpos = [ms.msg.position for ms in motors]
                qvel = [ms.msg.velocity for ms in motors]

                return JointState(
                    (*t, gs.timestamp),
                    (*hz, gs.hz),
                    (*qpos, gs.msg.value),
                    (*qvel, gs.msg.force),
                )

        raise TimeoutError

    def calibrate(self, joint_idx: int):
        t0 = time.perf_counter()

        while time.perf_counter() - t0 < self.timeout:
            if self.arm.calibrate_joint(joint_idx):
                return

            time.sleep(0.1)

        raise TimeoutError
