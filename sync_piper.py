import time

import numpy as np
import placo
from ischedule import run_loop, schedule
from placo_utils.visualization import robot_frame_viz, robot_viz

from piper_utils import Piper

robot = placo.RobotWrapper("piper", placo.Flags.ignore_collisions)

solver = placo.KinematicsSolver(robot)

solver.mask_fbase(True)
solver.enable_velocity_limits(True)

effector_task = solver.add_frame_task("joint6", np.eye(4))
effector_task.configure("joint6", "soft", 1.0, 1.0)

viz = robot_viz(robot)

t = 0
dt = 1 / 120
solver.dt = dt

piper = Piper("can0")

piper.reset()

time.sleep(1)


@schedule(interval=dt)
def loop():
    global t
    t += dt

    q = piper.recv_q()
    ee = piper.recv_ee()

    for i, pos in enumerate(q.pos):
        robot.set_joint(f"joint{i + 1}", pos)

    width = ee.width

    robot.set_joint("gripper", width)

    robot.set_joint("gripper_joint1", width / 2)
    robot.set_joint("gripper_joint2", -width / 2)

    robot.update_kinematics()

    # Displaying the robot, effector and target
    viz.display(robot.state.q)
    robot_frame_viz(robot, "gripper_tcp")


if __name__ == "__main__":
    run_loop()
