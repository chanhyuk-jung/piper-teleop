from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from functools import wraps
from queue import Queue

import numpy as np
import placo
from ischedule import run_loop, schedule
from numpy.typing import ArrayLike
from pyAgxArm.protocols.can_protocol.drivers import AgxGripperDriverDefault
from pyAgxArm.protocols.can_protocol.drivers.piper.default.driver import Driver
from pyAgxArm.protocols.can_protocol.msgs.core import MessageAbstract
from pyAgxArm.protocols.can_protocol.msgs.piper.default import ArmMsgFeedbackHighSpd
from scipy.spatial.transform import Rotation as R

from pyAgxArm import (
    AgxArmFactory,
    ArmModel,
    create_agx_arm_config,
)


def vr_to_flange(pos, quat):
    m = np.eye(4)

    x, y, z = pos
    m[:3, -1] = np.array([-z, -x, y])

    A = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]).T
    m[:3, :3] = A.T @ R.from_quat(quat).as_matrix() @ A

    return m


class Kinematics:
    robot: placo.RobotWrapper
    solver: placo.DynamicsSolver

    effector_task: placo.DynamicsFrameTask

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

        effector_task.T_world_frame = solver.robot.get_T_world_frame(effector_name)

        self.gripper_task = gripper_task = solver.add_joints_task()

        gripper_task.configure(gripper_name, "soft", 1)
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
        qpos = [self.robot.get_joint(f"joint{i + 1}") for i in range(6)]
        qpos.append(self.robot.get_joint(self.gripper_name))
        return tuple(qpos)

    def forward(self):
        return self.robot.get_T_world_frame(self.effector_name)

    def inverse(
        self,
        frame: ArrayLike,
        ee: float,
        *,
        vel: ArrayLike | None = None,
        ee_vel: float = 0.0,
    ):
        self.effector_task.T_world_frame = np.asarray(frame)
        self.gripper_task.set_joint(self.gripper_name, ee, ee_vel)

        if vel is not None:
            self.effector_task.position().dtarget_world = np.asarray(vel)

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


class RealPiper(threading.Thread):
    def __init__(self, channel, *, urdf_path, dt: float = 0.005, timeout: float = 1.0):
        super().__init__(daemon=True)

        self.timeout = timeout

        self.lock = threading.Lock()

        cfg = create_agx_arm_config(robot=ArmModel.PIPER, channel=channel)

        self.arm = arm = AgxArmFactory.create_arm(cfg)
        self.ee = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)

        arm.connect()

        time.sleep(0.1)

        self.enable_torque()

        arm.set_speed_percent(100)
        arm.set_motion_mode("j")

        self.k = Kinematics(urdf_path, dt=dt)

        self.move_qpos([0] * 7, timeout=10)

        self.q_pos_zero = self.k.get_qpos()

        self.goal_q = Queue(maxsize=1_000)

        self.robot_m = self.k.forward()
        self.prev_m = self.robot_m.copy()
        self.prev_gripper = 0.0

        self.ema_pos = self.prev_m[:3, -1].copy()

        self.anchor_m = None

    def enable_torque(self):
        t0 = time.perf_counter()

        while time.perf_counter() - t0 < self.timeout:
            info = self.arm.enable()

            if info:
                return

            time.sleep(0.01)

        raise TimeoutError

    def disable_torque(self):
        t0 = time.perf_counter()

        while time.perf_counter() - t0 < self.timeout:
            info = self.arm.disable()

            if info:
                return

            time.sleep(0.01)

        raise TimeoutError

    def move_qpos(self, qpos, force: float = 1.0, *, timeout: float = 0.0):
        qpos = np.asarray(qpos, dtype=np.float64)

        self.arm.move_j(list(qpos[:6]))
        self.ee.move_gripper_m(qpos[6], force)

        if timeout > 0:
            t0 = time.perf_counter()

            time.sleep(1 / 100)

            while time.perf_counter() - t0 < timeout:
                status = self.arm.get_arm_status()

                if status is None:
                    continue

                info = status.msg.motion_status

                if info == 0x00:
                    return

                time.sleep(0.01)
            else:
                raise TimeoutError

    def read_q(self):
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

    def close(self):
        self.arm.disconnect()

    def run(self):
        def ik_loop():
            with self.lock:
                if self.goal_q.empty():
                    return

                goal = self.goal_q.get()

            self.k.inverse(
                goal["frame"],
                goal["gripper"],
                vel=goal["vel"],
                ee_vel=goal["gripper_vel"],
            )

            qpos = self.k.get_qpos()
            self.move_qpos(qpos)

        schedule(ik_loop, interval=self.k.dt)

        run_loop()

    def init(self, pos, quat):
        q = self.read_q()
        self.k.set_qpos(np.array(q.pos) - self.q_pos_zero)

        self.robot_m = self.k.forward()
        self.prev_m = self.robot_m.copy()
        self.prev_gripper = q.pos[6]

        m = vr_to_flange(pos, quat)
        self.anchor_m = m.copy()

    def update_target(self, pos, quat, width, delta):
        if self.anchor_m is None:
            return

        m = vr_to_flange(pos, quat)

        delta_pos = m[:3, -1] - self.anchor_m[:3, -1]
        m[:3, -1] = delta_pos * 1.5 + self.robot_m[:3, -1]

        delta_rot = R.from_matrix(self.anchor_m[:3, :3]).inv() * R.from_matrix(
            m[:3, :3]
        )
        m_rot = R.from_matrix(self.robot_m[:3, :3]) * (delta_rot**1.5)

        m[:3, :3] = m_rot.as_matrix()

        alpha = 0.4
        self.ema_pos = m[:3, -1] * alpha + self.ema_pos * (1 - alpha)

        m[:3, -1] = self.ema_pos

        times = np.linspace(0, 1, max(int(delta / self.k.dt), 2))[1:]

        gripper = width ** (1 / 2) * self.k.gripper_max

        target_vel = (m[:3, -1] - self.prev_m[:3, -1]) / delta
        target_gripper_vel = (gripper - self.prev_gripper) / delta

        for t in times:
            target_m = placo.interpolate_frames(self.prev_m, m, t)

            target_gripper = (gripper - self.prev_gripper) * t + self.prev_gripper

            self.goal_q.put(
                {
                    "frame": target_m,
                    "vel": target_vel,
                    "gripper": target_gripper,
                    "gripper_vel": target_gripper_vel,
                }
            )

        self.prev_m = m.copy()
        self.prev_gripper = gripper

    def clear_tqrget_q(self):
        with self.lock:
            while True:
                try:
                    self.goal_q.get_nowait()
                except queue.Empty:
                    break
