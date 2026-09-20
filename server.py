import asyncio
import json
import threading
import time
from queue import Queue

import click
import numpy as np
import placo
from ischedule import run_loop, schedule
from scipy.spatial.transform import Rotation as R
from websockets.asyncio.server import ServerConnection, serve

from piper_utils import Kinematics, RealPiper

piper = RealPiper("can0")

time.sleep(1)

piper.enable_torque()

time.sleep(1)

piper.send_qpos([0] * 7)

time.sleep(4)

q = piper.recv_q()
q_pos_zero = np.array(q.pos)

k = Kinematics("piper")
k.dt = 0.005

goal_q = Queue(maxsize=-1)


@schedule(interval=k.dt)
def ik_loop():
    if goal_q.empty():
        return

    goal = goal_q.get()

    k.inverse(
        goal["frame"], goal["gripper"], vel=goal["vel"], ee_vel=goal["gripper_vel"]
    )

    piper.send_qpos(k.get_qpos())


def vr_to_flange(pos, quat):
    m = np.eye(4)

    x, y, z = pos
    m[:3, -1] = np.array([-z, -x, y])

    A = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]).T
    m[:3, :3] = A.T @ R.from_quat(quat).as_matrix() @ A

    return m


alpha = 0.4


async def handler(websocket: ServerConnection):
    robot_m = np.eye(4)
    anchor_m = np.eye(4)

    prev_m = np.eye(4)
    prev_gripper = 0

    ema_m = prev_m[:3, -1].copy()

    async for message in websocket:
        msg = json.loads(message)

        if msg["type"] == "start":
            q = piper.recv_q()

            k.set_qpos(np.array(q.pos) - q_pos_zero)

            robot_m = k.forward()

            payload = msg["payload"]
            m = vr_to_flange(payload["position"], payload["quaternion"])

            anchor_m = m.copy()
            prev_m = robot_m.copy()
            prev_gripper = q.pos[-1]

            ema_m = prev_m[:3, -1].copy()

        elif msg["type"] == "move":
            payload = msg["payload"]

            m = vr_to_flange(payload["position"], payload["quaternion"])

            delta_pos = m[:3, -1] - anchor_m[:3, -1]
            m[:3, -1] = delta_pos * 1.5 + robot_m[:3, -1]

            delta_rot = R.from_matrix(anchor_m[:3, :3]).inv() * R.from_matrix(m[:3, :3])
            m_rot = R.from_matrix(robot_m[:3, :3]) * (delta_rot**1.5)

            m[:3, :3] = m_rot.as_matrix()

            ema_m = m[:3, -1] * alpha + ema_m * (1 - alpha)

            m[:3, -1] = ema_m

            times = np.linspace(0, 1, int(msg["delta"] / k.dt))[1:]

            gripper = payload["gripper"] ** (1 / 2) * k.gripper_max

            target_vel = (m[:3, -1] - prev_m[:3, -1]) / msg["delta"]
            target_gripper_vel = (gripper - prev_gripper) / msg["delta"]

            for t in times:
                target_m = placo.interpolate_frames(prev_m, m, t)

                target_gripper = (gripper - prev_gripper) / msg[
                    "delta"
                ] * t + prev_gripper

                goal_q.put(
                    {
                        "frame": target_m,
                        "vel": target_vel,
                        "gripper": target_gripper,
                        "gripper_vel": target_gripper_vel,
                    }
                )

            prev_m = m.copy()
            prev_gripper = gripper

        elif msg["type"] == "stop":
            with goal_q.mutex:
                goal_q.queue.clear()

        elif msg["type"] == "reset":
            with goal_q.mutex:
                goal_q.queue.clear()

            piper.send_qpos([0] * 7)

            k.set_qpos([0] * 7)


async def async_serve(port: int):
    server = await serve(handler, host="0.0.0.0", port=port)
    print(f"websocket server running at http://127.0.0.1:{port}")

    await server.serve_forever()


@click.command()
@click.option("--port", default=65432, help="port to run server on")
def main(port: int):
    t = threading.Thread(target=run_loop, daemon=True)
    t.start()

    asyncio.run(async_serve(port))


if __name__ == "__main__":
    main()
