import asyncio
import json
import threading
from queue import Queue

import numpy as np
import placo
from ischedule import run_loop, schedule
from placo_utils.tf import tf
from placo_utils.visualization import frame_viz, get_viewer, robot_frame_viz, robot_viz
from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from websockets.asyncio.server import serve

from piper import PiperIK

HOST = "0.0.0.0"
PORT = 65432

dt = 0.005


def setup_robot(
    robot: PiperIK,
    *,
    dt: float = 0.005,
    pos_weight: float = 1e2,
    rot_weight: float = 1e-1,
):
    robot.solver.mask_fbase(True)

    robot.dt = dt
    robot.solver.enable_velocity_limits(True)

    robot.effector_task.configure(robot.effector_name, "soft", pos_weight, rot_weight)

    ik.set_goal(ik.get_frame())



def make_viz(robot: placo.RobotWrapper):
    viz = robot_viz(robot)

    vis = get_viewer()
    vis["/Cameras/default"].set_transform(tf.rotation_matrix(np.pi, [0, 0, 1]))
    vis["/Cameras/default/rotated/<object>"].set_property("zoom", 4.0)

    return viz


def render(viz, ik):
    viz.display(ik.q)

    robot_frame_viz(ik.robot, ik.effector_name)

    frame = ik.get_goal_frame()
    frame_viz("target", frame)


ik = PiperIK("piper")

setup_robot(ik, dt=dt)

viz = make_viz(ik.robot)

render(viz, ik)

"""
cfg = create_agx_arm_config(
    robot=ArmModel.PIPER, firmeware_version=PiperFW.DEFAULT, channel="can0"
)

arm = AgxArmFactory.create_arm(cfg)

end_effector = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)

arm.connect()

while not arm.enable():
    time.sleep(0.1)

arm.set_speed_percent(100)

time.sleep(0.1)

arm.move_j([0 for _ in range(6)])

end_effector.move_gripper_m(value=0.1, force=3.0)
"""

goal_q = Queue(maxsize=-1)


@schedule(interval=dt)
def ik_loop():
    goal = goal_q.get()

    ik.set_goal(goal["frame"], goal["gripper"])

    ik.solve()

    """
    joints, gripper = robot.get_joints()

    arm.move_j(state)
    end_effector.move_gripper_m(value=gripper, force=1.0)
    """


def vr_to_flange(pos, quat):
    m = np.eye(4)

    x, y, z = pos
    pos = np.array([-z, -x, y])

    m[:3, -1] = pos

    rotation = R.from_quat(quat)

    A = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]).T

    m[:3, :3] = A.T @ rotation.as_matrix() @ A

    return m


async def handler(websocket):
    robot_m = np.eye(4)
    anchor_m = np.eye(4)

    prev_m = np.eye(4)
    prev_open_length = 0

    prev_timestamp = 0

    async for message in websocket:
        msg = json.loads(message)

        if msg["type"] == "init_pose":
            """
            ja = arm.get_joint_angles()
            while ja is None:
                ja = arm.get_joint_angles()

            joints = [ja.msg[i] for i in range(6)]

            robot.set_joints(joints)
            """

            robot_m = ik.get_frame()

            payload = msg["payload"]

            pos = payload["position"]
            quat = payload["quaternion"]

            m = vr_to_flange(pos, quat)

            anchor_m = m.copy()

            prev_timestamp = msg["timestamp"]
            prev_m = robot_m.copy()
            prev_open_length = (1 - payload["gripper"]) ** 2 * 0.1

            render(viz, ik)

        elif msg["type"] == "pose":
            msg_dt = msg["timestamp"] - prev_timestamp
            prev_timestamp = msg["timestamp"]

            payload = msg["payload"]

            pos = payload["position"]
            quat = payload["quaternion"]

            open_length = (1 - payload["gripper"]) ** 2 * 0.1

            m = vr_to_flange(pos, quat)

            delta_pos = m[:3, -1] - anchor_m[:3, -1]
            m[:3, -1] = delta_pos + robot_m[:3, -1]

            max_d = 0.0015

            dis = m[:3, -1] - prev_m[:3, -1]
            mag = np.linalg.norm(dis)

            if mag > max_d:
                m[:3, -1] = dis / (mag + 1e-8) * max_d + prev_m[:3, -1]

            delta_rot = anchor_m[:3, :3].T @ m[:3, :3]
            m[:3, :3] = robot_m[:3, :3] @ delta_rot

            max_rot = 0.01

            rotation = R.from_matrix(prev_m[:3, :3].T @ m[:3, :3])
            mag = rotation.magnitude()

            if mag > max_rot:
                m[:3, :3] = (
                    prev_m[:3, :3]
                    @ R.from_rotvec(rotation.as_rotvec() / mag * max_rot).as_matrix()
                )

            times = np.linspace(0, 1, num=min(max(int(msg_dt / dt), 1), 10_000))

            target_ms = np.eye(4)[None, :].repeat(len(times), axis=0)

            target_ms[:, :3, -1] = (
                times[:, None] * (m[:3, -1] - prev_m[:3, -1]) + prev_m[:3, -1]
            )

            slerp = Slerp([0, 1], R.from_matrix(np.stack([prev_m[:3, :3], m[:3, :3]])))
            target_ms[:, :3, :3] = slerp(times).as_matrix()

            target_open_lengths = (
                open_length - prev_open_length
            ) * times + prev_open_length

            for target_m, target_open_length in zip(target_ms, target_open_lengths):
                goal_q.put({"frame": target_m, "gripper": target_open_length})

            prev_m = m.copy()
            prev_open_length = open_length

            if ik.effector_task.position().error_norm() > 0.05:
                event = {"event": "VIBRATE"}
                await websocket.send(json.dumps(event))

            render(viz, ik)

        elif msg["type"] == "stop":
            with goal_q.mutex:
                goal_q.queue.clear()


async def main():
    server = await serve(handler, host=HOST, port=PORT)
    await server.serve_forever()


if __name__ == "__main__":
    t = threading.Thread(target=run_loop, daemon=True)
    t.start()

    asyncio.run(main())
