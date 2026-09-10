import asyncio
import json
import threading
from queue import Queue

import click
import numpy as np
import placo
from ischedule import run_loop, schedule
from placo_utils.visualization import frame_viz, robot_frame_viz, robot_viz
from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config
from scipy.spatial.transform import Rotation as R
from websockets.asyncio.server import serve

# configs
dt = 0.008

effector_name = "gripper_tcp"
gripper_name = "gripper"

gripper_max = 0.1

pos_weight = 1.0
rot_weight = 1e-4

max_d = 0.01
max_rot = 0.2


robot = placo.RobotWrapper("piper")

# configure solver
solver = placo.KinematicsSolver(robot)

solver.dt = dt

solver.mask_fbase(True)
solver.enable_velocity_limits(True)
solver.enable_joint_limits(True)


# setup gripper
gear_task = solver.add_gear_task()
gear_task.configure("gear", "hard")

gear_task.set_gear("gripper_joint1", "gripper", 0.5)
gear_task.set_gear("gripper_joint2", "gripper", -0.5)


# setup ik task
effector_task = solver.add_frame_task(effector_name, np.eye(4))
effector_task.configure(effector_name, "soft", pos_weight, rot_weight)

gripper_task = solver.add_joints_task()
gripper_task.configure(gripper_name, "soft", 1.0)

effector_task.T_world_frame = robot.get_T_world_frame(effector_name)
gripper_task.set_joint(gripper_name, 0)


# setup ik regularization
regularization_task = solver.add_regularization_task(1e-4)


# setup visualization
viz = robot_viz(robot)

viz.display(robot.state.q)
robot_frame_viz(robot, effector_name)
frame_viz("target", effector_task.T_world_frame)


"""
# setup robot arm
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


# reset to home
arm.move_j([0 for _ in range(6)])
end_effector.move_gripper_m(value=0.0, force=3.0)
"""


goal_q = Queue(maxsize=-1)


@schedule(interval=dt)
def ik_loop():
    if goal_q.empty():
        return

    goal = goal_q.get()

    effector_task.T_world_frame = goal["frame"]
    gripper_task.set_joint(gripper_name, goal["gripper"])

    solver.solve(True)
    robot.update_kinematics()

    joints = [robot.get_joint(f"joint{i + 1}") for i in range(6)]
    gripper = robot.get_joint("gripper")

    """
    arm.move_j(joints)
    end_effector.move_gripper_m(value=gripper, force=1.0)
    """


def vr_to_flange(pos, quat):
    m = np.eye(4)

    x, y, z = pos
    m[:3, -1] = np.array([-z, -x, y])

    A = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]).T
    m[:3, :3] = A.T @ R.from_quat(quat).as_matrix() @ A

    return m


async def handler(websocket):
    robot_m = np.eye(4)
    anchor_m = np.eye(4)

    prev_m = np.eye(4)
    prev_gripper = 0

    async for message in websocket:
        msg = json.loads(message)

        if msg["type"] == "init_pose":
            """
            # get robot state
            ja = arm.get_joint_angles()
            gs = end_effector.get_gripper_status()

            while (ja is None) or (gs is None):
                if ja is None:
                    ja = arm.get_joint_angles()

                if gs is None:
                    gs = end_effector.get_gripper_status()

            joints = [ja.msg[i] for i in range(6)]
            gripper = max(gs.msg.value, 0) ** (1 / 2) * gripper_max

            # sync robot state
            for i, joint in enumerate(joints):
                solver.robot.set_joint(f"joint{i + 1}", joint)

            robot.set_joint("gripper", gripper)
            robot.set_joint("gripper_joint1", gripper / 2)
            robot.set_joint("gripper_joint2", -gripper / 2)

            robot.update_kinematics()
            """

            # fk robot frame
            robot_m = robot.get_T_world_frame(effector_name)

            payload = msg["payload"]
            m = vr_to_flange(payload["position"], payload["quaternion"])

            anchor_m = m.copy()
            prev_m = robot_m.copy()
            prev_gripper = robot.get_joint(gripper_name)

        elif msg["type"] == "pose":
            payload = msg["payload"]

            gripper = payload["gripper"] ** (1 / 2) * gripper_max
            m = vr_to_flange(payload["position"], payload["quaternion"])

            delta_pos = m[:3, -1] - anchor_m[:3, -1]
            m[:3, -1] = delta_pos + robot_m[:3, -1]

            # limit distance
            dis = m[:3, -1] - prev_m[:3, -1]
            mag = np.linalg.norm(dis)

            if mag > max_d:
                m[:3, -1] = dis / (mag + 1e-8) * max_d + prev_m[:3, -1]

            # limit rotation
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
            times = np.linspace(0, 1, num=int(msg["delta"] / dt))

            for t in times:
                target_m = placo.interpolate_frames(prev_m, m, t)
                target_gripper = (gripper - prev_gripper) * t + prev_gripper

                goal_q.put({"frame": target_m, "gripper": target_gripper})

            prev_m = m.copy()
            prev_gripper = gripper

        elif msg["type"] == "stop":
            with goal_q.mutex:
                goal_q.queue.clear()

        viz.display(robot.state.q)
        robot_frame_viz(robot, effector_name)
        frame_viz("target", effector_task.T_world_frame)


async def async_serve(port):
    server = await serve(handler, host="0.0.0.0", port=port)
    print(f"websocket server running at http://127.0.0.1:{port}")
    await server.serve_forever()


@click.command()
@click.option("--port", default=65432, help="port to run server on")
def main(port):
    stop_event = threading.Event()

    t = threading.Thread(target=run_loop, args=(stop_event,))
    t.start()

    asyncio.run(async_serve(port))

    stop_event.set()
    t.join()


if __name__ == "__main__":
    main()
