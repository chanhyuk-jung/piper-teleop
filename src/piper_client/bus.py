import time
from dataclasses import dataclass
from enum import IntEnum, unique
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyAgxArm import (
    AgxArmFactory,
    ArmModel,
    create_agx_arm_config,
)


@dataclass(frozen=True)
class Joints:
    timestamp: NDArray[np.float64]

    position: NDArray[np.float64]
    velocity: NDArray[np.float64]

    torque: NDArray[np.float64]
    current: NDArray[np.float64]


@dataclass(frozen=True)
class Gripper:
    timestamp: float

    width: float
    force: float


@unique
class ControlType(IntEnum):
    STANDBY = 0x00
    CAN = 0x01
    TEACHING = 0x02
    ETHERNET = 0x03
    WIFI = 0x04
    REMOTE = 0x05
    LINKAGE_TEACHING_INPUT = 0x06
    OFFLINE_TRAJECTORY = 0x07
    UNKNOWN = 0xFF


class ArmStatus(IntEnum):
    NORMAL = 0x00
    EMERGENCY_STOP = 0x01
    NO_SOLUTION = 0x02
    SINGULARITY_POINT = 0x03
    TARGET_POS_EXCEEDS_LIMIT = 0x04
    JOINT_COMMUNICATION_ERR = 0x05
    JOINT_BRAKE_NOT_RELEASED = 0x06
    COLLISION_OCCURRED = 0x07
    OVERSPEED_DURING_TEACHING_DRAG = 0x08
    JOINT_STATUS_ERR = 0x09
    OTHER_ERROR = 0x0A
    TEACHING_RECORD = 0x0B
    TEACHING_EXECUTION = 0x0C
    TEACHING_PAUSE = 0x0D
    MAIN_CONTROLLER_NTC_OVER_TEMPERATURE = 0x0E
    RELEASE_RESISTOR_NTC_OVER_TEMPERATURE = 0x0F
    UNKNOWN = 0xFF


@unique
class MoveType(IntEnum):
    POSITION = 0x00
    JOINT = 0x01
    LINE = 0x02
    CURVE = 0x03
    MIT = 0x04  # Piper firmware < v188; v188+ uses 0x06
    CPV = 0x05
    UNKNOWN = 0xFF


@unique
class TeachingStatus(IntEnum):
    DISABLED = 0x00  # Stop
    START_RECORDING = 0x01  # Start recording (enter drag-to-teach mode)
    STOP_RECORDING = 0x02  # Stop recording (exit drag-to-teach mode)
    EXECUTE_TRAJECTORY = 0x03  # Execute trajectory (replay recorded trajectory)
    PAUSE_EXECUTION = 0x04  # Pause execution
    RESUME_EXECUTION = 0x05  # Resume execution (continue trajectory replay)
    TERMINATE_EXECUTION = 0x06  # Terminate execution
    MOVE_TO_START = 0x07  # Move to trajectory start point


@unique
class MotionStatus(IntEnum):
    REACHED = 0x00
    MOVING = 0x01
    UNKNOWN = 0xFF


@dataclass
class Status:
    arm: ArmStatus
    control: ControlType
    move: MoveType

    motion: MotionStatus
    teaching: TeachingStatus

    comm_errors: dict[str, bool]
    limit_exceeds: dict[str, bool]

    def is_standby(self) -> bool | None:
        if self.control == ControlType.STANDBY:
            return True
        elif self.control == ControlType.CAN:
            return False

        return None

    def reached(self) -> bool | None:
        if self.move == MotionStatus.REACHED:
            return True
        elif self.move == MotionStatus.MOVING:
            return False

        return None

    def comm_ok(self) -> bool:
        return not any(self.comm_errors)

    def limit_exceeded(self) -> bool:
        return any(self.limit_exceeds)


class Bus:
    joint_names: tuple[str, ...] = (
        "base",
        "shoulder",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "tool_roll",
    )

    def __init__(
        self, channel: str, *, version: str = "default", timeout: float = 1.0
    ) -> None:
        self.timeout = timeout
        self.name_to_idx: dict[str, Literal[1, 2, 3, 4, 5, 6]] = {
            "base": 1,
            "shoulder": 2,
            "elbow": 3,
            "wrist_roll": 4,
            "wrist_pitch": 5,
            "tool_roll": 6,
        }

        version_map: dict[str, Literal["default", "v183", "v188", "v189"]] = {
            "default": "default",
            "v183": "v183",
            "v188": "v188",
            "v189": "v189",
        }

        cfg = create_agx_arm_config(
            robot=ArmModel.PIPER,
            channel=channel,
            firmeware_version=version_map[version],
        )

        self.arm = arm = AgxArmFactory.create_arm(cfg)
        self.ee = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)

    def connect(self) -> None:
        self.arm.connect()

        self.arm.set_speed_percent(percent=100)
        time.sleep(0.1)

    def disconnect(self) -> None:
        self.arm.disconnect()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.disconnect()
        return False

    def enable_control(self, percent: int = 100) -> None:
        if percent < 0 or percent > 100:
            raise ValueError

        self.arm.set_speed_percent(percent)

    def enable_torque(self) -> None:
        def enable():
            result = set()

            for name in self.joint_names:
                result.add(self.arm.enable(self.name_to_idx[name]))

            enabled = False not in result
            return enabled

        t0 = time.perf_counter()

        enabled = enable()
        while not enabled:
            if time.perf_counter() - t0 >= self.timeout:
                raise TimeoutError

            time.sleep(0.01)

            enabled = enable()

    def disable_torque(self) -> None:
        def disable():
            result = set()
            for name in self.joint_names:
                result.add(self.arm.disable(self.name_to_idx[name]))

            disabled = False not in result
            return disabled

        t0 = time.perf_counter()

        disabled = disable()
        while not disabled:
            if time.perf_counter() - t0 >= self.timeout:
                raise TimeoutError

            time.sleep(0.01)

            disabled = disable()

    def move_joints(self, joints: ArrayLike) -> None:
        joints = np.asarray(joints, dtype=np.float64)
        self.arm.move_j(list(joints))

    def move_mit(
        self,
        pos: ArrayLike,
        vel: ArrayLike | None = None,
        kp: float = 10.0,
        kd: float = 0.5,
        t_ff: float = 0.0,
    ):
        pos = np.asarray(pos)

        if vel is None:
            vel = np.zeros_like(pos)
        else:
            vel = np.asarray(vel)

        for i, name in enumerate(self.joint_names):
            self.arm.move_mit(self.name_to_idx[name], pos[i], vel[i], kp, kd, t_ff)

    def read_joints(self) -> Joints:
        timestamp: list[float] = []
        position: list[float] = []
        velocity: list[float] = []
        torque: list[float] = []
        current: list[float] = []

        def read():
            motor_states = [
                self.arm.get_motor_states(self.name_to_idx[name])
                for name in self.joint_names
            ]

            for ms in motor_states:
                if ms is None:
                    return None

                timestamp.append(ms.timestamp)

                msg = ms.msg

                position.append(msg.position)
                velocity.append(msg.velocity)
                torque.append(msg.torque)
                current.append(msg.current)

            def float64(x: ArrayLike):
                return np.asarray(x, dtype=np.float64)

            return Joints(
                float64(timestamp),
                float64(position),
                float64(velocity),
                float64(torque),
                float64(current),
            )

        t0 = time.perf_counter()

        joints = read()
        while joints is None:
            if time.perf_counter() - t0 >= self.timeout:
                raise TimeoutError

            time.sleep(0)

            joints = read()

        return joints

    def move_gripper(
        self,
        width: float,
        *,
        force: float = 1.0,
    ) -> None:
        self.ee.move_gripper_m(width, force)

    def read_gripper(self) -> Gripper:
        def read():
            gs = self.ee.get_gripper_status()

            if gs is None:
                return None

            msg = gs.msg

            return Gripper(
                timestamp=gs.timestamp,
                width=msg.value,
                force=msg.force,
            )

        t0 = time.perf_counter()
        gripper = read()
        while gripper is None:
            if time.perf_counter() - t0 >= self.timeout:
                raise TimeoutError

            time.sleep(0)

            gripper = read()

        return gripper

    def read_status(self) -> Status:
        def read():
            status = self.arm.get_arm_status()

            if status is None:
                return None

            msg = status.msg

            error = msg.err_status

            comm_errors = {
                name: getattr(error, f"communication_status_joint_{i + 1}")
                for i, name in enumerate(self.joint_names)
            }
            limit_exceeds = {
                name: getattr(error, f"joint_{i + 1}_angle_limit")
                for i, name in enumerate(self.joint_names)
            }

            return Status(
                ArmStatus(msg.arm_status),
                ControlType(msg.ctrl_mode),
                MoveType(msg.mode_feedback),
                MotionStatus(msg.motion_status),
                TeachingStatus(msg.teach_status),
                comm_errors,
                limit_exceeds,
            )

        t0 = time.perf_counter()

        status = read()
        while status is None:
            if time.perf_counter() - t0 >= self.timeout:
                raise TimeoutError

            time.sleep(0)

            status = read()

        return status

    def calibrate(self, name: str) -> None:
        self.arm.calibrate_joint(self.name_to_idx[name])

    def reset(self):
        self.arm.reset()
