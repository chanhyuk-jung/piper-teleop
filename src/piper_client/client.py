import queue
import threading
import time
from dataclasses import dataclass
from functools import wraps
from queue import Queue

import numpy as np
import placo
from ischedule import run_loop, schedule
from numpy.typing import ArrayLike, NDArray
from pyAgxArm import (
    AgxArmFactory,
    ArmModel,
    create_agx_arm_config,
)
from pyAgxArm.protocols.can_protocol.msgs.core import MessageAbstract
from pyAgxArm.protocols.can_protocol.msgs.piper.default import ArmMsgFeedbackHighSpd
from scipy.spatial.transform import Rotation as R


def vr_to_flange(pos, quat) -> NDArray[np.float64]:
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
        dt: float = 0.008,
        pos_weight: float = 1.0,
        rot_weight: float = 5e-3,
    ):
        self.effector_name: str = effector_name
        self.dt: float = dt

        self.robot = robot = placo.RobotWrapper(urdf_path)
        self.solver = solver = placo.DynamicsSolver(robot)

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
        posture.set_joints({f"joint{i + 1}": 0.0 for i in range(6)})
        posture.configure("posture", "soft", 1e-4)

    def set_qpos(self, joints: ArrayLike):
        joints = np.asarray(joints, dtype=np.float64)

        for i, joint in enumerate(joints[:6]):
            self.robot.set_joint(f"joint{i + 1}", joint)

        self.robot.update_kinematics()

    def get_qpos(self):
        qpos = [self.robot.get_joint(f"joint{i + 1}") for i in range(6)]
        return qpos

    def forward(self):
        return self.robot.get_T_world_frame(self.effector_name)

    def inverse(
        self,
        frame: ArrayLike,
        *,
        vel: ArrayLike | None = None,
    ):
        self.effector_task.T_world_frame = np.asarray(frame)

        if vel is not None:
            self.effector_task.position().dtarget_world = np.asarray(vel)

        result = self.solver.solve(True)
        self.robot.update_kinematics()

        return result


class JointKinematics:
    robot: placo.RobotWrapper
    solver: placo.DynamicsSolver

    def __init__(
        self,
        urdf_path: str,
        *,
        gripper_name: str = "gripper",
        dt: float = 0.008,
        gripper_max: float = 0.1,
    ):
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
        gears.set_gear("gripper_joint1", "gripper", 0.5)
        gears.add_gear("gripper_joint2", "gripper", -0.5)
        gears.configure("gear", "hard")

        self.joints_task = joints_task = solver.add_joints_task()

        joints_task.configure("joints", "soft", 1)

    def inverse(self, joints, *, vel=None):
        self.joints_task.set_joints(joints)

        if vel is not None:
            self.joints_task.set_joints_velocities(vel)

        result = self.solver.solve(True)
        self.robot.update_kinematics()

        return result

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


class Client:
    def __init__(self, channel: str, *, timeout: float = 1.0):
        self.timeout = timeout

        cfg = create_agx_arm_config(robot=ArmModel.PIPER, channel=channel)

        self.arm = arm = AgxArmFactory.create_arm(cfg)
        self.ee = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)

        arm.connect()

        time.sleep(0.1)

        arm.set_speed_percent(100)
        arm.set_motion_mode("j")

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


class EMA:
    def __init__(self, x, *, alpha: float = 0.4):
        self.ema = x
        self.alpha = alpha

    def __call__(self, x):
        self.ema = x * self.alpha + self.ema * (1 - self.alpha)
        return self.ema


class Planner:
    def __init__(self, frame, qpos, dt):
        self.frame = frame.copy()
        self.gripper = qpos[6]

        self.qpos = qpos
        self.dt = dt

        self.dt = dt

    def plan_ee(self, frame, gripper, dt):
        target_vel = (frame[:3, -1] - self.frame[:3, -1]) / dt
        target_gripper_vel = (gripper - self.gripper) / dt

        times = np.linspace(0, 1, max(int(dt / self.dt), 2))[1:]

        trajectory = []
        for t in times:
            target_m = placo.interpolate_frames(self.frame, frame, t)
            target_gripper = (gripper - self.gripper) * t + self.gripper
            yield {
                "frame": target_m,
                "vel": target_vel,
                "ee": target_gripper,
                "ee_vel": target_gripper_vel,
            }

        self.frame = frame.copy()
        self.gripper = gripper

        return trajectory

    def plan_qpos(self, qpos, dt):
        target_vel = (np.array(qpos) - self.qpos) / dt

        times = np.linspace(0, 1, max(int(dt / self.dt), 2))[1:]

        trajectory = []
        for t in times:
            target_qpos = target_vel * t + self.qpos
            yield {"qpos": target_qpos, "vel": target_vel}

        self.qpos = qpos.copy()

        return trajectory


class CoordTransform:
    robot_home: NDArray[np.float64]
    vr_home: NDArray[np.float64]

    def __init__(self, robot, vr):
        self.robot_home = robot.copy()
        self.vr_home = vr.copy()

    def __call__(self, frame):
        delta_pos = frame[:3, -1] - self.vr_home[:3, -1]
        frame[:3, -1] = delta_pos * 1.5 + self.robot_home[:3, -1]

        delta_rot = R.from_matrix(self.vr_home[:3, :3]).inv() * R.from_matrix(
            frame[:3, :3]
        )
        m_rot = R.from_matrix(self.robot_home[:3, :3]) * (delta_rot**1.5)

        frame[:3, :3] = m_rot.as_matrix()

        return frame


class TeleopThread(threading.Thread):
    def __init__(self, channel, *, urdf_path, dt: float = 0.005, timeout: float = 1.0):
        super().__init__()

        self.client = Client(channel, timeout=timeout)

        self.client.enable_torque()
        self.client.move_qpos([0] * 7, timeout=10)

        self.k = Kinematics(urdf_path, dt=dt)
        self.jk = JointKinematics(urdf_path, dt=dt)

        self.home = self.k.forward().copy()

        self.action_q = Queue(maxsize=1_000)

        self.coord_tsfm = None
        self.ema = None
        self.planner = None
        self.j_planner = None

        self.action = tuple([0.0] * 7)

        self.timeout = timeout
        self.lock = threading.Lock()

        self.multiplier = 2

    def run(self):
        def ik_loop():
            with self.lock:
                if self.action_q.empty():
                    return

                qpos = self.action_q.get()

            self.client.move_qpos(qpos)

        schedule(ik_loop, interval=self.k.dt)

        run_loop()

    def start_thread(self):
        self.daemon = True
        self.start()

    def stop_thread(self):
        pass

    def init(self, pos, quat):
        q = self.client.read_q()
        qpos = np.array(q.pos)

        self.k.set_qpos(qpos[:6])
        qpos[6] /= self.multiplier
        self.jk.set_qpos(qpos)

        robot = self.k.forward()
        vr = vr_to_flange(pos, quat)

        self.coord_tsfm = CoordTransform(robot, vr)

        self.ema = EMA(self.k.forward()[:3, -1])

        self.planner = Planner(robot, self.jk.get_qpos(), self.k.dt)

    def vr_to_robot(self, pos, quat):
        if self.coord_tsfm is None:
            return

        m = vr_to_flange(pos, quat)
        m = self.coord_tsfm(m)

        return m

    def sync_ks(self):
        qpos = self.k.get_qpos()
        self.jk.set_qpos(qpos[:6])
        self.k.robot.set_joint(self.jk.gripper_name, self.jk.get_qpos()[6])

    def update_ee(self, pos, quat, width, delta):
        if self.ema is None:
            raise RuntimeError

        if self.planner is None:
            raise RuntimeError

        if self.coord_tsfm is None:
            return

        frame = vr_to_flange(pos, quat)
        frame = self.coord_tsfm(frame)

        frame[:3, -1] = self.ema(frame[:3, -1])

        gripper = width ** (1 / 2) * self.jk.gripper_max / self.multiplier

        self.planner.frame = self.k.forward().copy()
        self.planner.gripper = float(self.jk.get_qpos()[6])

        for target in self.planner.plan_ee(frame, gripper, delta):
            self.k.inverse(
                target["frame"],
                vel=target["vel"],
            )
            self.jk.inverse(
                {self.jk.gripper_name: target["ee"]},
                vel={self.jk.gripper_name: target["ee_vel"]},
            )

            qpos = list(self.k.get_qpos())
            qpos.append(self.jk.get_qpos()[6] * self.multiplier)
            qpos = tuple(qpos)

            self.action_q.put(qpos)

            self.action = qpos

        self.sync_ks()

    def update_qpos(self, qpos, delta):
        if self.planner is None:
            raise RuntimeError

        self.planner.qpos = self.jk.get_qpos()

        for target in self.planner.plan_qpos(qpos, delta):
            qpos = target["qpos"]

            joints = {f"joint{i + 1}": qpos[i] for i in range(6)}
            joints[self.jk.gripper_name] = qpos[6]

            vel = target["vel"]

            joint_vels = {f"joint{i + 1}": vel[i] for i in range(6)}
            joint_vels[self.jk.gripper_name] = vel[6]

            self.jk.inverse(
                joints,
                vel=joint_vels,
            )

            qpos = self.jk.get_qpos()

            self.action_q.put(qpos)

            self.action = qpos

    def pause(self):
        with self.lock:
            while True:
                try:
                    self.action_q.get_nowait()
                except queue.Empty:
                    break

    def reset(self):
        self.pause()

        self.client.move_qpos([0] * 7)
        self.k.set_qpos([0] * 7)
