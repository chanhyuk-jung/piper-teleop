import json
import time
from collections import deque
from threading import Event, Lock
from typing import cast

import numpy as np
from agx_bus import AgilexBus, AgilexGripper
from ischedule import run_loop, schedule
from numpy.typing import NDArray
from websockets import ServerConnection

from .camera import CameraThread, open_camera
from .dynamics import Dynamics
from .planning import EffectorPlanner, JointPlanner
from .quest import QuestThread, quest_to_flange
from .recording import DatasetWriter, Image, RecordThread


def serve_teleop(
    data_path: str,
    can: str,
    urdf: str,
    task_name: str,
    gripper_name: str,
    wrist_cam: int,
    front_cam: int,
    port=4000,
    hz: int = 90,
    multiplier: int = 5,
    ema: float = 0.5,
    gripper_ema: float = 0.2,
    shard_size: int = 1024,
    bufsize: int = 128,
    iters: int = 8,
):
    dt = 1 / (hz * multiplier)

    bus = AgilexBus(can, name="piper", version="default")
    ee = AgilexGripper(bus)

    bus.connect()

    bus.enable_torque()

    bus.move_joints([0.0] * 6)
    ee.move_gripper(width=0.0)

    state = bus.read_state()
    while not state.reached:
        time.sleep(0.01)
        state = bus.read_state()

    solver = Dynamics(
        urdf,
        effector_name=task_name,
        gripper_name=gripper_name,
        dt=dt / iters,
        iters=iters,
    )

    task_planner = EffectorPlanner(dt, alpha=ema)
    q_planner = JointPlanner(dt, alpha=1.0, max_vel=0.5)

    gripper_planner = JointPlanner(dt, alpha=gripper_ema)

    wrist_cap = open_camera(wrist_cam)
    front_cap = open_camera(front_cam)

    wrist = CameraThread(wrist_cap)
    front = CameraThread(front_cap)

    wrist.start_thread()
    front.start_thread()

    writer = DatasetWriter(data_path, shard_size=shard_size, buffer_size=bufsize)
    record_thread = RecordThread(writer)

    record_thread.start_thread()

    q = deque(maxlen=multiplier)
    lock = Lock()

    msg_q = deque(maxlen=32)

    cancel_event = Event()

    quest = QuestThread(port)

    robot_home = np.eye(4)

    @quest.subscribe("start")
    def update_home(msg):
        nonlocal robot_home

        joints = bus.read_joints()
        gripper = ee.read_gripper()

        solver.set_joints(joints.position)

        payload = msg["payload"]

        robot_home = solver.forward()

        task_planner.set_start(robot_home)
        quest.set_anchor(quest_to_flange(payload["position"], payload["quaternion"]))

        gripper_planner.set_start(np.array(gripper.width))

        cancel_event.clear()

    @quest.subscribe("follow")
    def follow(msg):
        msg_q.append(msg)

    @quest.subscribe("pause")
    def pause(_):
        cancel_event.set()

        with lock:
            q.clear()

        record_thread.cancel()

        bus.move_joints([0.0] * 6)
        ee.move_gripper(width=0.0)

    @quest.subscribe("go_home")
    def go_home(msg):
        msg_q.append(msg)

    @quest.subscribe("save")
    def save(_):
        record_thread.save()

    @quest.async_subscribe("waiting")
    async def idle(ws: ServerConnection, _):
        if record_thread.ready():
            await ws.send(json.dumps({"type": "ready"}))

    quest.start_thread()
    print("server started")

    stop_control = Event()

    @schedule(interval=1 / hz)
    def record_loop():
        t0 = time.perf_counter()
        joints = bus.read_joints()
        gripper = ee.read_gripper()

        state = {
            "qpos": np.append(joints.position, gripper.width),
            "qvel": np.append(joints.velocity, gripper.force),
        }

        wrist_img = cast(NDArray, wrist.latest_frame)
        front_img = cast(NDArray, front.latest_frame)

        obs = {}
        obs.update(state)

        obs["timestamp"] = np.array(t0)

        obs["wrist_img"] = Image(wrist_img)
        obs["front_img"] = Image(front_img)

        if len(msg_q) == 0 or cancel_event.is_set():
            return

        msg = msg_q.popleft()

        if msg["type"] == "follow":
            payload = msg["payload"]

            pose = quest_to_flange(payload["position"], payload["quaternion"])

            delta_pos = pose[:3, -1] - quest.anchor[:3, -1]
            pos = robot_home[:3, -1] + delta_pos

            delta_rot = quest.anchor[:3, :3].T @ pose[:3, :3]
            rot = robot_home[:3, :3] @ delta_rot

            task_frame = np.eye(4)

            task_frame[:3, -1] = pos
            task_frame[:3, :3] = rot

            gripper = payload["gripper"]

            gripper = gripper ** (1 / 2) * 0.1

            task_traj = task_planner.plan(task_frame, dt=msg["delta"])
            gripper_traj = gripper_planner.plan(np.array(gripper), dt=msg["delta"])

            for (waypoint, vel), gripper_t in zip(task_traj, gripper_traj, strict=True):
                qpos = solver.inverse(waypoint, vel=vel)

                qpos = np.append(qpos[:6], gripper_t)
                q.append(qpos)

            action = {"qpos": qpos}

        elif msg["type"] == "go_home":
            qpos = np.asarray(solver.get_joints()[:7], dtype=np.float64)
            home = np.zeros(7)

            delta = home - qpos
            end = qpos + delta

            q_planner.set_start(qpos)
            for waypoint in q_planner.plan(end, msg["delta"]):
                q.append(waypoint)

            action = {"qpos": waypoint}
            solver.set_joints(waypoint)

        obs["dt"] = np.array(time.perf_counter() - t0)

        record_thread.record(obs, action)

    @schedule(interval=dt)
    def control_loop():
        with lock:
            if len(q) == 0:
                return

            action = q.popleft()

        bus.move_joints(action[:6])
        ee.move_gripper(width=action[6])

    try:
        run_loop(stop_control)

    except KeyboardInterrupt:
        print("closing server...")

        stop_control.set()

        wrist.stop_thread()
        front.stop_thread()
        quest.stop_thread()

        bus.move_joints([0.0] * 6)
        ee.move_gripper(width=0.0)

        state = bus.read_state()
        while not state.reached:
            time.sleep(0.01)
            state = bus.read_state()

        record_thread.stop_thread()
        bus.disconnect()


def server_cli():
    import tyro

    tyro.cli(serve_teleop)
