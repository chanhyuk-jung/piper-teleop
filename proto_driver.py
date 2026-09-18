import time
from collections.abc import Sequence
from dataclasses import dataclass
from functools import wraps

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyAgxArm import (
    AgxArmFactory,
    ArmModel,
    create_agx_arm_config,
    resolve_firmware_profile,
)
from pyAgxArm.protocols.can_protocol.drivers.core.arm_driver_abstract import (
    ArmDriverAbstract,
)
from pyAgxArm.protocols.can_protocol.msgs.core import MessageAbstract
from pyAgxArm.protocols.can_protocol.msgs.piper.default import ArmMsgFeedbackHighSpd


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


class Piper:
    def __init__(self, channel="can0", timeout: float = 1.0):
        cfg = create_agx_arm_config(robot=ArmModel.PIPER, channel=channel)

        self.robot = robot = AgxArmFactory.create_arm(cfg)
        self.ee = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)

        robot.connect()

        if robot.has_comm_error():
            raise RuntimeError

        robot.set_speed_percent(100)
        robot.set_motion_mode("j")

        self.timeout: float = timeout

    def close(self):
        self.robot.disconnect()

    def enable_torque(self):
        t0 = time.perf_counter()

        while time.perf_counter() - t0 < self.timeout:
            if self.robot.enable():
                return

        raise TimeoutError

    def disable_torque(self):
        t0 = time.perf_counter()

        while time.perf_counter() - t0 < self.timeout:
            if self.robot.disable():
                return

        raise TimeoutError

    def send_qpos(self, qpos: Sequence[float], force: float = 1.0):
        self.robot.move_j(list(qpos[:-1]))
        self.ee.move_gripper_m(qpos[-1], force)

    def recv_q(self) -> JointState:
        t0 = time.perf_counter()

        while time.perf_counter() - t0 < self.timeout:
            motors: list[MessageAbstract[ArmMsgFeedbackHighSpd]] = []

            for i in range(6):
                if i == 0:
                    ms = self.robot.get_motor_states(1)
                elif i == 1:
                    ms = self.robot.get_motor_states(2)
                elif i == 2:
                    ms = self.robot.get_motor_states(3)
                elif i == 3:
                    ms = self.robot.get_motor_states(4)
                elif i == 4:
                    ms = self.robot.get_motor_states(5)
                elif i == 5:
                    ms = self.robot.get_motor_states(6)
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
