import time

import numpy as np
import placo
from ischedule import run_loop, schedule
from placo_utils.tf import tf
from placo_utils.visualization import frame_viz, robot_frame_viz, robot_viz
from scipy.spatial.transform import Rotation as R

from piper_utils import Piper

robot = placo.RobotWrapper("piper", placo.Flags.ignore_collisions)

solver = placo.KinematicsSolver(robot)

solver.mask_fbase(True)
solver.enable_velocity_limits(True)

effector_task = solver.add_frame_task("gripper_tcp", np.eye(4))
effector_task.configure("joint6", "soft", 1.0, 1.0)

viz = robot_viz(robot)

t = 0
dt = 0.005
solver.dt = dt

piper = Piper("can0")

for i in range(6):
    robot.set_joint(f"joint{i + 1}", 0)

robot.set_joint("gripper", 0)

robot.set_joint("gripper_joint1", 0)
robot.set_joint("gripper_joint2", 0)

robot.update_kinematics()

piper.send_qpos([0] * 6)
piper.send_ee(0)

time.sleep(4)

m = tf.translation_matrix([0.4, 0.2 * np.sin(t), 0.2])

rotation = R.from_euler("xyz", [0, -180, 0], degrees=True)
m[:3, :3] = rotation.as_matrix()

effector_task.T_world_frame = m

viz.display(robot.state.q)
robot_frame_viz(robot, "gripper_tcp")
frame_viz("target", effector_task.T_world_frame)


@schedule(interval=dt)
def loop():
    global t
    t += dt

    m = tf.translation_matrix([0.4, 0.2 * np.sin(2 * t), 0.2])

    rotation = R.from_euler("xyz", [0, -180, 0], degrees=True)
    m[:3, :3] = rotation.as_matrix()

    effector_task.T_world_frame = m

    solver.solve(True)
    robot.update_kinematics()

    state = []
    for i in range(6):
        rads = robot.get_joint(f"joint{i + 1}")
        degs = np.rad2deg(rads)
        mili_degs = degs * 1000
        state.append(int(mili_degs))

    piper.send_qpos([robot.get_joint(f"joint{i + 1}") for i in range(6)])
    piper.send_ee(0)

    viz.display(robot.state.q)
    robot_frame_viz(robot, "gripper_tcp")
    frame_viz("target", effector_task.T_world_frame)


run_loop()
