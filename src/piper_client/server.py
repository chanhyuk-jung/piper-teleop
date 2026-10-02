import time
from queue import Queue
from threading import Event, Lock

import click
import numpy as np
from ischedule import run_loop, schedule

from .camera import CameraThread, open_camera
from .dynamics import Dynamics
from .planning import EffectorPlanner, JointPlanner
from .quest import QuestThread, quest_to_flange
from .recording import RecordThread, ZarrRecorder
from .robots import PiperFollower


@click.command()
@click.option("--data-path", type=str, required=True)
@click.option("--port", default=4000)
@click.option("--can", default="can0")
@click.option("--urdf", default="piper")
@click.option("--task_name", default="gripper_tcp")
@click.option("--gripper_name", default="gripper")
@click.option("--dt", default=1 / 450)
@click.option("--ema", default=0.1)
@click.option("--wrist-cam", type=int, required=True)
@click.option("--front-cam", type=int, required=True)
def main(
    data_path: str,
    port=4000,
    can: str = "can0",
    urdf: str = "piper",
    task_name: str = "gripper_tcp",
    gripper_name: str = "gripper",
    dt: float = 1 / 450,
    ema: float = 0.1,
    wrist_cam: int = 2,
    front_cam: int = 0,
):
    robot = PiperFollower(can, dt=dt)
    solver = Dynamics(urdf, effector_name=task_name, gripper_name=gripper_name, dt=dt)
    ee_planner = EffectorPlanner(dt, alpha=ema)
    q_planner = JointPlanner(dt, alpha=1.0)

    wrist_cap = open_camera(wrist_cam)
    front_cap = open_camera(front_cam)

    wrist = CameraThread(wrist_cap)
    front = CameraThread(front_cap)

    wrist.start_thread()
    front.start_thread()

    print("cameras started")

    recorder = ZarrRecorder(path=data_path)
    record_thread = RecordThread(recorder)

    record_thread.start_thread()

    print("recorder started")

    q = Queue(maxsize=-1)
    msg_q = Queue(maxsize=-1)

    sync_lock = Lock()

    quest = QuestThread(port)

    robot_home = np.eye(4)

    @quest.subscribe("start")
    def update_home(msg):
        nonlocal robot_home

        qpos = robot.get_state()["qpos"]
        solver.set_joints(qpos[:6])

        payload = msg["payload"]

        robot_home = solver.forward()
        ee_planner.set_start(robot_home)
        quest.set_anchor(quest_to_flange(payload["position"], payload["quaternion"]))

    @quest.subscribe("idle")
    def idle(msg):
        if sync_lock.locked():
            sync_lock.release()

    @quest.subscribe("follow")
    def follow(msg):
        msg_q.put(msg)

    @quest.subscribe("pause")
    def pause(_):
        while not q.empty():
            q.get_nowait()

    @quest.subscribe("go_home")
    def go_home(msg):
        msg_q.put(msg)

    @quest.subscribe("save")
    def save(_):
        record_thread.save()

    quest.start_thread()
    print("server started")

    stop_control = Event()

    @schedule(interval=1 / 90)
    def record_loop():
        while sync_lock.locked():
            time.sleep(1e-6)

        if msg_q.empty():
            return

        t = time.perf_counter()
        state = robot.get_state()

        wrist_img = wrist.latest_frame
        front_img = front.latest_frame

        obs = {}
        obs["timestamp"] = np.array([t], dtype=np.float64)
        obs.update(state)

        obs["wrist_img"] = wrist_img
        obs["front_img"] = front_img

        msg = msg_q.get()
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

            ee = gripper ** (1 / 2) * 0.1

            for waypoint, vel in ee_planner.plan(task_frame, msg["delta"]):
                qpos = solver.inverse(waypoint, vel=vel)

                q.put(np.append(qpos[:6], ee))

            qpos = solver.get_joints()
            action = {"qpos": np.append(qpos[:6], ee)}

            record_thread.record(obs, action)

        elif msg["type"] == "go_home":
            qpos = np.asarray(solver.get_joints()[:7], dtype=np.float64)
            home = np.zeros(7)

            delta = home - qpos
            end = qpos + delta

            q_planner.set_start(qpos)
            for waypoint in q_planner.plan(end, msg["delta"]):
                q.put(waypoint)

            action = {"qpos": waypoint}
            solver.set_joints(waypoint)

            record_thread.record(obs, action)

    @schedule(interval=dt)
    def control_loop():
        if q.empty():
            return

        action = q.get()
        robot.move(action)

    try:
        run_loop(stop_control)
    except KeyboardInterrupt:
        print("closing server...")
        stop_control.set()

        wrist.stop_thread()
        front.stop_thread()
        quest.stop_thread()
        record_thread.stop_thread()

        record_thread.join()


if __name__ == "__main__":
    main()
