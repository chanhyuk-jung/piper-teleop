from queue import Queue

import h5py
from ischedule import run_loop, schedule

from .client import Client

client = Client("can0")

client.move_qpos([0] * 7, timeout=10)

f = h5py.File("demo.hdf5", "r")

actions = f["data"]["demo_0"]["action"]["qpos"][:]

q = Queue(maxsize=-1)

for qpos in actions:
    q.put(qpos)


@schedule(interval=1 / 60)
def loop():
    qpos = q.get()
    client.move_qpos(qpos)


run_loop()
