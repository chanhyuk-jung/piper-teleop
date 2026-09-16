from __future__ import annotations

import functools
import time
from dataclasses import KW_ONLY, dataclass, field

import numpy as np
import placo
from numpy.typing import ArrayLike, NDArray

from piper_sdk import C_PiperInterface_V2 as PiperBus

STANDBY = 0x00
CAN_CONTROL = 0x01

MOVE_J = 0x01

DISABLE = 0x00
ENABLE = 0x01

CLEAR_AND_DISABLE = 0x02

EMERGENCY_STOP = 0x01
RESUME_EMERGENCY_STOP = 0x02
DISABLE_TRAJECTORY_CONTROL = 0x00
DISABLE_TEACH = 0x00


def clip(x: int, min_val: int, max_val: int) -> int:
    return min(max(x, min_val), max_val)


class PiperError(Exception):
    pass


class PiperNotOpenError(PiperError):
    pass


@dataclass(frozen=True)
class GripperState:
    timestamp: float
    hz: float

    width: float
    torque: float


@dataclass(frozen=True)
class JointState:
    timestamp: float
    hz: float
    pos: NDArray[np.float64]
    vel: NDArray[np.float64]
    torque: NDArray[np.float64]


def require_open(method):
    @functools.wraps(method)
    def _impl(self, *args, **kwargs):
        if not self.open:
            raise PiperNotOpenError

        output = method(self, *args, **kwargs)
        return output

    return _impl


def require_torque(method):
    @functools.wraps(method)
    def _impl(self, *args, **kwargs):
        if not self.torque:
            raise PiperError

        output = method(self, *args, **kwargs)
        return output

    return _impl


class Piper:
    open: bool = False
    torque: bool = False
    speed: int = 100

    def __init__(self, can: str, timeout: float = 1.0):
        self.bus: PiperBus = PiperBus(can_name=can)

        self.bus.ConnectPort()
        self.open = True

    @require_open
    def set_speed(self, speed: int = 50):
        self.speed = speed
        self.bus.MotionCtrl_2(CAN_CONTROL, MOVE_J, self.speed)

    @require_open
    def enable_torque(self, timeout: float = 1.0):
        self.bus.EnableArm()

        t0 = time.perf_counter()
        while not all(self.recv_arm_enable_status()):
            if time.perf_counter() - t0 >= timeout:
                raise TimeoutError("timeout exceeded while trying to enable arm")

            self.bus.EnableArm()
            time.sleep(0.1)

        ee = self.recv_ee()

        width = max(-0.1, min(ee.width, 0.1)) * 1_000_000
        torque = max(0, min(ee.torque, 5.0)) * 1_000

        while not self.bus.GetArmGripperMsgs().gripper_state.foc_status.driver_enable_status:
            if time.perf_counter() - t0 >= timeout:
                raise TimeoutError("timeout exceeded while trying to enable gripper")

            self.bus.GripperCtrl(int(width), int(torque), ENABLE)
            time.sleep(0.1)

        self.bus.MotionCtrl_2(CAN_CONTROL, MOVE_J, self.speed)

        self.torque = True

    @require_open
    def disable_torque(self, timeout: float = 1.0):
        self.bus.MotionCtrl_2(STANDBY, MOVE_J)

        self.bus.DisableArm()

        t0 = time.perf_counter()
        while not any(self.recv_arm_enable_status()):
            if time.perf_counter() - t0 >= timeout:
                raise TimeoutError("timeout exceeded while trying to disable arm")

            self.bus.DisableArm()
            time.sleep(0.1)

        ee = self.recv_ee()

        width = max(-0.1, min(ee.width, 0.1)) * 1_000_000
        torque = max(0, min(ee.torque, 5.0)) * 1_000

        while (
            self.bus.GetArmGripperMsgs().gripper_state.foc_status.driver_enable_status
        ):
            if time.perf_counter() - t0 >= timeout:
                raise TimeoutError("timeout exceeded while trying to disable gripper")

            self.bus.GripperCtrl(int(width), int(torque), DISABLE)
            time.sleep(0.1)

        self.bus.MotionCtrl_2(CAN_CONTROL, MOVE_J, self.speed)

        self.torque = False

    @require_open
    def recv_arm_enable_status(self) -> tuple[bool, ...]:
        msg = self.bus.GetArmLowSpdInfoMsgs()

        motors = (
            msg.motor_1,
            msg.motor_2,
            msg.motor_3,
            msg.motor_4,
            msg.motor_5,
            msg.motor_6,
        )

        enable_status = tuple(motor.foc_status.driver_enable_status for motor in motors)
        return enable_status

    @require_open
    def reset(self):
        # clears any errors or e-stops or teaching modes, moves on second move command
        self.bus.MotionCtrl_1(
            RESUME_EMERGENCY_STOP, DISABLE_TRAJECTORY_CONTROL, DISABLE_TEACH
        )

        self.bus.GripperCtrl(0, 0, CLEAR_AND_DISABLE)

    @require_open
    def emergency_stop(self):
        # ignores move commands & disables torque
        self.bus.MotionCtrl_1(EMERGENCY_STOP)

    @require_open
    def unlock(self):
        # unlocks emergency stop, moves on second move command
        self.bus.MotionCtrl_1(RESUME_EMERGENCY_STOP)

    @require_torque
    @require_open
    def send_qpos(self, qpos: ArrayLike):
        qpos = np.asarray(qpos, dtype=np.float64)
        qpos = np.rad2deg(qpos)

        j: tuple[int, ...] = tuple((qpos * 1000).astype(np.int32))

        self.bus.JointCtrl(
            joint_1=j[0],
            joint_2=j[1],
            joint_3=j[2],
            joint_4=j[3],
            joint_5=j[4],
            joint_6=j[5],
        )

    @require_open
    def recv_q(self) -> JointState:
        msg = self.bus.GetArmHighSpdInfoMsgs()

        timestamp = msg.time_stamp
        hz = msg.Hz

        motors = (
            msg.motor_1,
            msg.motor_2,
            msg.motor_3,
            msg.motor_4,
            msg.motor_5,
            msg.motor_6,
        )

        qpos = np.array([motor.pos for motor in motors]) / 1_000
        qvel = np.array([motor.motor_speed for motor in motors]) / 1_000
        torque = np.array([motor.effort for motor in motors]) / 1_000

        q = JointState(timestamp, hz, qpos, qvel, torque)

        return q

    @require_torque
    @require_open
    def send_ee(self, width: float, torque: float = 1.0):
        self.bus.GripperCtrl(int(width * 1_000_000), int(torque * 1_000), ENABLE)

    @require_open
    def recv_ee(self) -> GripperState:
        msg = self.bus.GetArmGripperMsgs()

        timestamp = msg.time_stamp
        hz = msg.Hz

        state = msg.gripper_state
        width = state.grippers_angle / 1_000_000
        torque = state.grippers_effort / 1_000

        ee = GripperState(timestamp, hz, width, torque)
        return ee

    @require_open
    def close(self):
        self.bus.DisconnectPort()
        self.open = False


@dataclass
class Kinematics:
    urdf_path: str

    _: KW_ONLY
    effector_name: str = "gripper_tcp"
    gripper_name: str = "gripper"
    dt: float = 0.008
    pos_weight: float = 1.0
    rot_weight: float = 5e-3
    gripper_max: float = 0.1

    robot: placo.RobotWrapper = field(init=False)
    solver: placo.DynamicsSolver = field(init=False)

    effector_task: placo.DynamicsFrameTask = field(init=False)
    gripper_task: placo.DynamicsJointsTask = field(init=False)

    def __post_init__(self):
        self.robot = robot = placo.RobotWrapper(self.urdf_path)

        self.solver = solver = placo.DynamicsSolver(robot)

        solver.dt = self.dt

        solver.enable_joint_limits(True)
        solver.enable_velocity_limits(True)

        solver.mask_fbase(True)

        gears = solver.add_gear_task()
        gears.add_gear("gripper_joint1", "gripper", 0.5)
        gears.add_gear("gripper_joint2", "gripper", -0.5)
        gears.configure("gear", "hard")

        self.effector_task = effector_task = solver.add_frame_task(
            self.effector_name, np.eye(4)
        )
        effector_task.configure(
            self.effector_name, "soft", self.pos_weight, self.rot_weight
        )
        effector_task.position().kp

        self.gripper_task = gripper_task = solver.add_joints_task()
        gripper_task.configure(self.gripper_name, "soft", 1.0)

        effector_task.T_world_frame = solver.robot.get_T_world_frame(self.effector_name)
        gripper_task.set_joint(self.gripper_name, 0)

        posture = solver.add_joints_task()
        posture.set_joints({f"joint{i + 1}": 0.0 for i in range(6)})
        posture.configure("posture", "soft", 1e-4)

    def set_qpos(self, joints):
        for i, joint in enumerate(joints):
            self.robot.set_joint(f"joint{i + 1}", joint)

        self.robot.update_kinematics()

    def get_qpos(self):
        joints = [self.robot.get_joint(f"joint{i + 1}") for i in range(6)]
        return joints

    def set_ee(self, meters: float):
        self.robot.set_joint("gripper", meters)
        self.robot.set_joint("gripper_joint1", meters / 2)
        self.robot.set_joint("gripper_joint2", -meters / 2)

        self.robot.update_kinematics()

    def get_ee(self):
        gripper = self.robot.get_joint(self.gripper_name)
        return gripper

    def forward(self):
        return self.robot.get_T_world_frame(self.effector_name)

    def inverse(self, frame, gripper: float, *, vel=None, ee_vel: float = 0):
        self.effector_task.T_world_frame = frame
        if vel is not None:
            self.effector_task.position().dtarget_world = vel

        self.gripper_task.set_joint(self.gripper_name, gripper, ee_vel)

        self.solver.solve(True)
        self.robot.update_kinematics()
