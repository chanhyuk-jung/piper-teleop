import numpy as np
import placo
from ischedule import run_loop, schedule
from placo_utils.tf import tf
from placo_utils.visualization import frame_viz, robot_frame_viz, robot_viz
from scipy.spatial.transform import Rotation as R

# Loading the robot
robot = placo.RobotWrapper("piper", placo.Flags.ignore_collisions)

# Creating the solver
solver = placo.KinematicsSolver(robot)

solver.mask_fbase(True)
solver.enable_velocity_limits(True)

effector_task = solver.add_frame_task("gripper_tcp", np.eye(4))
effector_task.configure("gripper_tcp", "soft", 1.0, 0.01)

viz = robot_viz(robot)

t = 0
dt = 0.01
solver.dt = dt


@schedule(interval=dt)
def loop():
    global t
    t += dt

    m = tf.translation_matrix([0.4, 0.2 * np.sin(t), 0.2])

    rotation = R.from_euler("xyz", [0, -180, 0], degrees=True)
    m[:3, :3] = rotation.as_matrix()

    effector_task.T_world_frame = m

    # Solving the IK
    solver.solve(True)
    robot.update_kinematics()

    # Displaying the robot, effector and target
    viz.display(robot.state.q)
    robot_frame_viz(robot, "gripper_tcp")
    frame_viz("target", effector_task.T_world_frame)


run_loop()
