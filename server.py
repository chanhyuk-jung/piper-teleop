import asyncio
import json
import multiprocessing as mp
import threading
import time
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

HOST = "0.0.0.0"
PORT = 65432

enable_collisions = False

robot = placo.RobotWrapper("piper")

for i in range(6):
    robot.set_joint(f"joint{i + 1}", 0)

robot.set_joint("gripper", 0.1)
robot.set_joint("gripper_joint1", 0.05)
robot.set_joint("gripper_joint2", -0.05)

robot.update_kinematics()

solver = placo.KinematicsSolver(robot)

dt = 0.005
solver.dt = dt

solver.mask_fbase(True)
solver.enable_velocity_limits(True)

gear_task = solver.add_gear_task()
gear_task.configure("gear", "hard")

gear_task.set_gear("gripper_joint1", "gripper", 0.5)
gear_task.set_gear("gripper_joint2", "gripper", -0.5)

joints_task = solver.add_joints_task()
joints_task.configure("gripper", "soft", 1.0)

effector_task = solver.add_frame_task("flange_joint", np.eye(4))
effector_task.configure("flange_joint", "soft", 1e2, 0.1)

joints_task = solver.add_joints_task()
joints_task.set_joints({f"joint{i + 1}": 0.0 for i in range(6)})
joints_task.configure("joints_regularization", "soft", 1e-5)

if enable_collisions:
    # Enabling self collisions avoidance
    avoid_self_collisions = solver.add_avoid_self_collisions_constraint()
    avoid_self_collisions.configure("avoid_self_collisions", "hard")

    # The constraint starts existing when contacts are 3cm away, and keeps a 1cm margin
    avoid_self_collisions.self_collisions_margin = 0.01  # [m]
    avoid_self_collisions.self_collisions_trigger = 0.03  # [m]

viz = robot_viz(robot)

vis = get_viewer()
vis["/Cameras/default"].set_transform(tf.rotation_matrix(np.pi, [0, 0, 1]))
vis["/Cameras/default/rotated/<object>"].set_property("zoom", 4.0)

effector_task.T_world_frame = robot.get_T_world_frame("flange_joint")

viz.display(robot.state.q)
robot_frame_viz(robot, "flange_joint")
frame_viz("target", effector_task.T_world_frame)

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

target_queue = Queue(maxsize=-1)


@schedule(interval=dt)
def ik_loop():
    target = target_queue.get()

    effector_task.T_world_frame = target["frame"]
    joints_task.set_joint("gripper", target["gripper"])

    solver.solve(True)
    robot.update_kinematics()

    """
    state = []
    for i in range(6):
        state.append(robot.get_joint(f"joint{i + 1}"))

    arm.move_j(state)
    end_effector.move_gripper_m(value=robot.get_joint("gripper"), force=1.0)
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
    robot_m = robot.get_T_world_frame("flange_joint")
    anchor_m = np.eye(4)

    prev_m = robot_m.copy()
    prev_open_length = robot.get_joint("gripper")

    prev_timestamp = 0

    async for message in websocket:
        msg = json.loads(message)

        if msg["type"] == "init_pose":
            """
            ja = arm.get_joint_angles()
            while ja is None:
                ja = arm.get_joint_angles()

            for i in range(6):
                robot.set_joint(f"joint{i + 1}", ja.msg[i])

            robot.update_kinematics()
            """

            robot_m = robot.get_T_world_frame("flange_joint")

            payload = msg["payload"]

            pos = payload["position"]
            quat = payload["quaternion"]

            m = vr_to_flange(pos, quat)

            anchor_m = m.copy()

            prev_timestamp = msg["timestamp"]
            prev_m = robot_m.copy()
            prev_open_length = (1 - payload["gripper"]) ** 2 * 0.1

            viz.display(robot.state.q)
            robot_frame_viz(robot, "flange_joint")
            frame_viz("target", robot_m)

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
                target_queue.put({"frame": target_m, "gripper": target_open_length})

            prev_m = m.copy()
            prev_open_length = open_length

            if effector_task.position().error_norm() > 0.05:
                event = {"event": "VIBRATE"}
                await websocket.send(json.dumps(event))

            viz.display(robot.state.q)
            robot_frame_viz(robot, "flange_joint")
            frame_viz("target", m)

        elif msg["type"] == "stop":
            with target_queue.mutex:
                target_queue.queue.clear()


async def main():
    server = await serve(handler, host=HOST, port=PORT)
    await server.serve_forever()


if __name__ == "__main__":
    t = threading.Thread(target=run_loop, daemon=True)
    t.start()

    asyncio.run(main())
