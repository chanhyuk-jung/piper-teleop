import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from piper_sdk import C_PiperInterface_V2 as PiperBus

NO_CONTROL = 0x00
CAN_CONTROL = 0x01

JOINT_MODE = 0x01

GRIPPER_DISABLE = 0x00
GRIPPER_ENABLE = 0x01

EMERGENCY_STOP = 0x01
RESUME_EMERGENCY_STOP = 0x02
DISABLE_TRAJECTORY_CONTROL = 0x00
DISABLE_TEACH = 0x00


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


class Piper:
    def __init__(self, can: str, write_only: bool = False, timeout: float = 1.0):
        self.bus: PiperBus = PiperBus(can_name=can)

        self.bus.ConnectPort(start_thread=not write_only)
        time.sleep(0.1)

        self.reset()
        time.sleep(0.1)

        self.enable_torque(timeout)
        self.enable_control()

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

    def enable_torque(self, timeout: float = 1.0):
        self.bus.EnableArm()

        t0 = time.perf_counter()
        while not all(self.recv_arm_enable_status()):
            if time.perf_counter() - t0 >= timeout:
                raise TimeoutError("timeout exceeded while trying to enable arm")

            self.bus.EnableArm()
            time.sleep(0.1)

        msg = self.bus.GetArmGripperMsgs()

        g = msg.gripper_state
        pos = g.grippers_angle
        torque = g.grippers_effort

        enabled = g.foc_status.driver_enable_status

        t0 = time.perf_counter()
        while not enabled:
            if time.perf_counter() - t0 >= timeout:
                raise TimeoutError("timeout exceeded while trying to enable gripper")

            self.bus.GripperCtrl(pos, torque, GRIPPER_ENABLE)
            time.sleep(0.1)

            msg = self.bus.GetArmGripperMsgs()
            enabled = msg.gripper_state.foc_status.driver_enable_status

    def disable_torque(self, timeout: float = 1.0):
        self.bus.DisableArm()

        t0 = time.perf_counter()
        while not any(self.recv_arm_enable_status()):
            if time.perf_counter() - t0 >= timeout:
                raise TimeoutError("timeout exceeded while trying to disable arm")

            self.bus.DisableArm()
            time.sleep(0.1)

        msg = self.bus.GetArmGripperMsgs()

        g = msg.gripper_state
        pos = g.grippers_angle
        torque = g.grippers_effort

        enabled = g.foc_status.driver_enable_status

        t0 = time.perf_counter()
        while enabled:
            if time.perf_counter() - t0 >= timeout:
                raise TimeoutError("timeout exceeded while trying to disable gripper")

            self.bus.GripperCtrl(pos, torque, GRIPPER_DISABLE)
            time.sleep(0.1)

            msg = self.bus.GetArmGripperMsgs()
            enabled = msg.gripper_state.foc_status.driver_enable_status

    def enable_control(self):
        self.bus.MotionCtrl_2(CAN_CONTROL, JOINT_MODE)

    def disable_control(self):
        self.bus.MotionCtrl_2(NO_CONTROL)

    def set_speed(self, speed: int):
        speed = 100
        self.bus.ModeCtrl(0x01, 0x01, speed, 0x00)

    def close(self):
        self.bus.DisconnectPort()

    def reset(self):
        self.bus.MotionCtrl_1(
            RESUME_EMERGENCY_STOP, DISABLE_TRAJECTORY_CONTROL, DISABLE_TEACH
        )

    def emergency_stop(self):
        self.bus.MotionCtrl_1(EMERGENCY_STOP)

    def unlock(self):
        self.bus.EmergencyStop(RESUME_EMERGENCY_STOP)

    def send_qpos(self, qpos: ArrayLike):
        qpos = np.asarray(qpos, dtype=np.float64)

        assert len(qpos) == 6, (
            "qpos must have 6 or 7 joints where 7 includes the gripper"
        )

        qpos = np.rad2deg(qpos) * 1_000

        j: list[int] = list(qpos.astype(np.int32))

        self.bus.JointCtrl(
            joint_1=j[0],
            joint_2=j[1],
            joint_3=j[2],
            joint_4=j[3],
            joint_5=j[4],
            joint_6=j[5],
        )

    def send_ee(self, width: float, force: float = 1.0):
        self.bus.GripperCtrl(int(width * 1_000_000), int(force * 1_000), GRIPPER_ENABLE)

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

    def recv_ee(self) -> GripperState:
        msg = self.bus.GetArmGripperMsgs()

        timestamp = msg.time_stamp
        hz = msg.Hz

        state = msg.gripper_state
        width = state.grippers_angle / 1_000_000
        torque = state.grippers_effort / 1_000

        ee = GripperState(timestamp, hz, width, torque)

        return ee


if __name__ == "__main__":
    piper = Piper("can0")

    piper.send_qpos([0] * 6)
    piper.send_ee(0)

    time.sleep(4)

    q = piper.recv_q()
    ee = piper.recv_ee()
