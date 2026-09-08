import numpy as np
import placo
from ischedule import run_loop, schedule
from placo_utils.tf import tf
from placo_utils.visualization import frame_viz, robot_frame_viz, robot_viz
from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config
from scipy.spatial.transform import Rotation as R

"""
6axis robot reaching a given target (introduction example).
"""

# Loading the robot
robot = placo.RobotWrapper("piper", placo.Flags.ignore_collisions)

# Creating the solver
solver = placo.KinematicsSolver(robot)

solver.mask_fbase(True)
solver.enable_velocity_limits(True)

effector_task = solver.add_frame_task("joint6", np.eye(4))
effector_task.configure("joint6", "soft", 1.0, 1.0)

viz = robot_viz(robot)

t = 0
dt = 0.005
solver.dt = dt

cfg = create_agx_arm_config(
    robot=ArmModel.PIPER, firmeware_version=PiperFW.DEFAULT, channel="can0"
)

arm = AgxArmFactory.create_arm(cfg)
end_effector = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)

arm.connect()

arm.disable()


# @schedule(interval=dt)
def solve():
    global t
    t += dt

    m = tf.translation_matrix([0.3, 0.2 * np.sin(t), 0.2])

    rotation = R.from_euler("xyz", [0, -180, 0], degrees=True)
    m[:3, :3] = rotation.as_matrix()

    effector_task.T_world_frame = m

    # Solving the IK
    solver.solve(True)
    robot.update_kinematics()

    # Displaying the robot, effector and target
    viz.display(robot.state.q)
    robot_frame_viz(robot, "joint6")
    frame_viz("target", effector_task.T_world_frame)


@schedule(interval=dt)
def loop():
    global t
    t += dt

    ja = arm.get_joint_angles()
    gs = end_effector.get_gripper_status()

    for i in range(6):
        robot.set_joint(f"joint{i + 1}", ja.msg[i])

    robot.set_joint("gripper_joint1", gs.msg.value / 2)
    robot.set_joint("gripper_joint2", -gs.msg.value / 2)

    robot.update_kinematics()

    # Displaying the robot, effector and target
    viz.display(robot.state.q)
    robot_frame_viz(robot, "joint6")


joint_names = robot.joint_names()
print(list(joint_names))

run_loop()
