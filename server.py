import asyncio
import json
import threading
import time
from queue import Queue

import click
import numpy as np
import placo
from ischedule import run_loop, schedule
from placo_utils.visualization import frame_viz, robot_frame_viz, robot_viz
from scipy.spatial.transform import Rotation as R
from websockets.asyncio.server import ServerConnection, serve

from piper_utils import Kinematics, Piper

max_d = 0.01 * 1000
max_rot = 0.2 * 1000
ALPHA = 0.5


k = Kinematics("piper")
k.dt = 0.008

viz = robot_viz(k.robot)

piper = Piper("can0")

piper.enable_torque()

time.sleep(1)

piper.send_qpos([0] * 6)
piper.send_ee(0, 0)

time.sleep(4)

q_pos_zero = piper.recv_q().pos

viz.display(k.robot.state.q)
robot_frame_viz(k.robot, k.effector_name)
frame_viz("target", k.effector_task.T_world_frame)

goal_q = Queue(maxsize=-1)

data_q = Queue(maxsize=-1)


@schedule(interval=k.dt)
def ik_loop():
    if goal_q.empty():
        return

    goal = goal_q.get()

    k.inverse(goal["frame"], goal["dframe"], goal["gripper"], goal["dgripper"])

    joints = k.get_qpos()
    gripper = k.get_ee()

    q = piper.recv_q()
    ee = piper.recv_ee()

    state = {"q": q, "ee": ee}
    action = {"qpos": joints, "width": gripper}

    data_q.put((action, state))

    piper.send_qpos(joints)
    piper.send_ee(gripper, 1.0)


def vr_to_flange(pos, quat):
    m = np.eye(4)

    x, y, z = pos
    m[:3, -1] = np.array([-z, -x, y])

    A = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]).T
    m[:3, :3] = A.T @ R.from_quat(quat).as_matrix() @ A

    return m


async def handler(websocket: ServerConnection):
    robot_m = np.eye(4)
    anchor_m = np.eye(4)

    prev_m = np.eye(4)
    prev_gripper = 0

    async for message in websocket:
        msg = json.loads(message)

        if msg["type"] == "start":
            q = piper.recv_q()
            ee = piper.recv_ee()

            k.set_qpos(q.pos - q_pos_zero)
            k.set_ee(ee.width)

            robot_m = k.forward()

            payload = msg["payload"]
            m = vr_to_flange(payload["position"], payload["quaternion"])

            anchor_m = m.copy()
            prev_m = robot_m.copy()
            prev_gripper = k.get_ee()

        elif msg["type"] == "move":
            payload = msg["payload"]

            m = vr_to_flange(payload["position"], payload["quaternion"])

            delta_pos = m[:3, -1] - anchor_m[:3, -1]
            m[:3, -1] = delta_pos + robot_m[:3, -1]

            dis = m[:3, -1] - prev_m[:3, -1]
            mag = np.linalg.norm(dis)

            if mag > max_d:
                m[:3, -1] = dis / (mag + 1e-8) * max_d + prev_m[:3, -1]

            delta_rot = R.from_matrix(anchor_m[:3, :3]).inv() * R.from_matrix(m[:3, :3])
            m_rot = R.from_matrix(robot_m[:3, :3]) * delta_rot

            prev_rot = R.from_matrix(prev_m[:3, :3])
            delta_rot = prev_rot.inv() * m_rot

            mag = delta_rot.magnitude()
            if mag > max_rot:
                scale = float(max_rot / mag)
                m_rot = prev_rot * (delta_rot**scale)

            m[:3, :3] = m_rot.as_matrix()

            # interpolate frames
            times = np.linspace(0, 1, num=int(msg["delta"] / k.dt))

            gripper = payload["gripper"] ** (1 / 2) * k.gripper_max

            target_vel = (m[:3, -1] - prev_m[:3, -1]) / msg["delta"]

            for t in times:
                target_m = placo.interpolate_frames(prev_m, m, t)

                target_gripper = (gripper - prev_gripper) * t + prev_gripper
                target_gripper_vel = (gripper - prev_gripper) / msg["delta"]

                goal_q.put({"frame": target_m, "dframe": target_vel, "gripper": target_gripper, "dgripper": target_gripper_vel})

            prev_m = m.copy()
            prev_gripper = gripper

        elif msg["type"] == "stop":
            with goal_q.mutex:
                goal_q.queue.clear()

        elif msg["type"] == "reset":
            with goal_q.mutex:
                goal_q.queue.clear()

            piper.send_qpos([0] * 6)
            piper.send_ee(0, 0)

            k.set_qpos([0] * 6)
            k.set_ee(0)

        viz.display(k.robot.state.q)
        robot_frame_viz(k.robot, k.effector_name)
        frame_viz("target", k.effector_task.T_world_frame)


async def async_serve(port):
    server = await serve(handler, host="0.0.0.0", port=port)
    print(f"websocket server running at http://127.0.0.1:{port}")

    await server.serve_forever()


@click.command()
@click.option("--port", default=65432, help="port to run server on")
def main(port):
    stop_event = threading.Event()

    t = threading.Thread(target=run_loop, args=(stop_event,), daemon=True)
    t.start()

    asyncio.run(async_serve(port))


if __name__ == "__main__":
    main()
