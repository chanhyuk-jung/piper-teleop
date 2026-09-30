import numpy as np

from piper_client.bus import Bus

bus = Bus("can0")

bus.connect()

input("press enter to disable torque")
bus.disable_torque()

joints_min = None
joints_max = None

history = []

while True:
    joints = bus.read_joints()
    qpos = joints.position

    history.append(qpos)

    joints_max = np.max(np.stack(history), axis=0)
    joints_min = np.min(np.stack(history), axis=0)

    history = [joints_max, joints_min]

    print(
        f"joint max values: {joints_max.tolist()}, joint min values: {joints_min.tolist()}"
    )
