import asyncio
import json
import threading
from queue import Queue

import numpy as np
import placo
from ischedule import run_loop, schedule
from placo_utils.visualization import frame_viz, get_viewer, robot_frame_viz, robot_viz
from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from websockets.asyncio.server import serve

HOST = "0.0.0.0"
PORT = 65432

dt = 0.005

effector_name = "flange_joint"
gripper_name = "gripper"
gripper_max = 0.1

pos_weight = 1e2
rot_weight = 1e-1


robot = placo.RobotWrapper("piper")

solver = placo.KinematicsSolver(robot)

solver.mask_fbase(True)

solver.dt = dt
solver.enable_velocity_limits(True)

gear_task = solver.add_gear_task()
gear_task.configure("gear", "hard")

gear_task.set_gear("gripper_joint1", "gripper", 0.5)
gear_task.set_gear("gripper_joint2", "gripper", -0.5)

effector_task = solver.add_frame_task(effector_name, np.eye(4))
gripper_task = solver.add_joints_task()

effector_task.configure(effector_name, "soft", pos_weight, rot_weight)

effector_task.T_world_frame = robot.get_T_world_frame(effector_name)
gripper_task.set_joint(gripper_name, 0)


viz = robot_viz(robot)

vis = get_viewer()
vis["/Cameras/default/rotated/<object>"].set_property("zoom", 4.0)


viz.display(robot.state.q)

robot_frame_viz(robot, effector_name)

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

goal_q = Queue(maxsize=-1)


@schedule(interval=dt)
def ik_loop():
    goal = goal_q.get()

    effector_task.T_world_frame = goal["frame"]
    gripper_task.set_joint(gripper_name, goal["gripper"])

    solver.solve(True)
    robot.update_kinematics()

    """
    joints = [robot.get_joint(f"joint{i + 1}") for i in range(6)]
    gripper = robot.get_joint("gripper")

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

            gs = end_effector.get_gripper_status()
            gripper = gs.msg.value

            for i, joint in enumerate(joints):
                solver.robot.set_joint(f"joint{i + 1}", joint)

            if gripper is not None:
                assert 0.1 >= gripper >= 0
                robot.set_joint("gripper", gripper)
                robot.set_joint("gripper_joint1", gripper / 2)
                robot.set_joint("gripper_joint2", -gripper / 2)

            robot.update_kinematics()
            """

            robot_m = robot.get_T_world_frame(effector_name)
            prev_open_length = robot.get_joint(gripper_name)

            payload = msg["payload"]

            pos = payload["position"]
            quat = payload["quaternion"]

            m = vr_to_flange(pos, quat)

            anchor_m = m.copy()

            prev_timestamp = msg["timestamp"]
            prev_m = robot_m.copy()

            viz.display(robot.state.q)

            robot_frame_viz(robot, effector_name)

            frame_viz("target", effector_task.T_world_frame)

        elif msg["type"] == "pose":
            msg_dt = msg["timestamp"] - prev_timestamp
            prev_timestamp = msg["timestamp"]

            payload = msg["payload"]

            pos = payload["position"]
            quat = payload["quaternion"]

            open_length = payload["gripper"] ** 2 * gripper_max

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

            if effector_task.position().error_norm() > 0.05:
                event = {"event": "VIBRATE"}
                await websocket.send(json.dumps(event))

            viz.display(robot.state.q)

            robot_frame_viz(robot, effector_name)

            frame_viz("target", effector_task.T_world_frame)

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
