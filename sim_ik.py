import numpy as np
import placo
from ischedule import run_loop, schedule
from placo_utils.tf import tf
from placo_utils.visualization import frame_viz, robot_frame_viz, robot_viz
from scipy.spatial.transform import Rotation as R

from piper_utils import Kinematics

k = Kinematics("piper")

k.set_qpos([0] * 6)
k.set_ee(0)

viz = robot_viz(k.robot)

t = 0
dt = 0.01
k.dt = dt


@schedule(interval=dt)
def loop():
    global t
    t += dt

    m = tf.translation_matrix([0.4, 0.2 * np.sin(t), 0.2])
    vel = np.array([0, 0.2 * np.cos(t), 0])

    rotation = R.from_euler("xyz", [0, -180, 0], degrees=True)
    m[:3, :3] = rotation.as_matrix()

    k.inverse(m, 0, vel=vel)

    viz.display(k.robot.state.q)
    robot_frame_viz(k.robot, k.effector_name)
    frame_viz("target", k.effector_task.T_world_frame)


run_loop()
