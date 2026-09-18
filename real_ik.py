import time

import numpy as np
from ischedule import run_loop, schedule
from placo_utils.tf import tf
from placo_utils.visualization import frame_viz, robot_frame_viz, robot_viz
from scipy.spatial.transform import Rotation as R

from piper_utils import Kinematics
from proto_driver import Piper

k = Kinematics("piper")

viz = robot_viz(k.robot)

t = 0
dt = 0.005

k.dt = dt

piper = Piper("can0")

time.sleep(1)

piper.enable_torque()

time.sleep(1)

piper.send_qpos([0] * 7)
k.set_qpos([0] * 7)

time.sleep(4)

m = tf.translation_matrix([0.4, 0.2 * np.sin(t), 0.2])

rotation = R.from_euler("xyz", [0, -180, 0], degrees=True)
m[:3, :3] = rotation.as_matrix()

k.effector_task.T_world_frame = m

viz.display(k.robot.state.q)
robot_frame_viz(k.robot, "gripper_tcp")
frame_viz("target", k.effector_task.T_world_frame)


@schedule(interval=dt)
def loop():
    global t
    t += dt

    m = tf.translation_matrix([0.4, 0.2 * np.sin(t), 0.2])

    rotation = R.from_euler("xyz", [0, -180, 0], degrees=True)
    m[:3, :3] = rotation.as_matrix()

    k.effector_task.T_world_frame = m

    k.inverse(m, 0.0)

    qpos = k.get_qpos()
    piper.send_qpos(qpos)

    viz.display(k.robot.state.q)
    robot_frame_viz(k.robot, k.gripper_name)
    frame_viz("target", k.effector_task.T_world_frame)


run_loop()
