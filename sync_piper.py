import time

from ischedule import run_loop, schedule
from placo_utils.visualization import robot_frame_viz, robot_viz

from piper_utils import Kinematics, RealPiper

k = Kinematics("piper")

viz = robot_viz(k.robot)

dt = 1 / 120

piper = RealPiper("can0")

time.sleep(1)

piper.disable_torque()

time.sleep(1)


@schedule(interval=dt)
def loop():
    q = piper.recv_q()

    k.set_qpos(q.pos)

    viz.display(k.robot.state.q)
    robot_frame_viz(k.robot, k.gripper_name)


if __name__ == "__main__":
    run_loop()
