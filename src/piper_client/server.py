import asyncio
import json
import multiprocessing as mp
import os

import click
import h5py
import numpy as np
from websockets.asyncio.server import ServerConnection, serve

from .camera import CameraThread, open_camera
from .client import TeleopThread


async def async_serve(port: int, wrist_cam, front_cam, dataset):
    teleop = TeleopThread("can0", urdf_path="piper", dt=0.002)
    teleop.start_thread()

    record = Recorder(teleop, dataset, wrist_cam, front_cam)
    record.start_process()

    async def handler(websocket: ServerConnection):
        async for message in websocket:
            msg = json.loads(message)
            payload = msg["payload"]

            if msg["type"] == "start":
                teleop.init(payload["position"], payload["quaternion"])

            elif msg["type"] == "follow":
                obs = record.get_obs()

                teleop.update_ee(
                    payload["position"],
                    payload["quaternion"],
                    payload["gripper"],
                    msg["delta"],
                )

                action = {"qpos": teleop.action}

                record.add(obs, action)

            elif msg["type"] == "pause":
                teleop.pause()

            elif msg["type"] == "go_home":
                obs = record.get_obs()

                clamped = []
                for theta in obs["qpos"]:
                    if abs(theta) < 0.05:
                        theta = 0

                    clamped.append(theta)

                if np.abs(clamped).sum() != 0:
                    teleop.update_qpos([0.0] * 7, msg["delta"])

                action = {"qpos": teleop.action}

                record.add(obs, action)

            elif msg["type"] == "save":
                record.flush()

    server = await serve(handler, host="0.0.0.0", port=port)
    print(f"websocket server running at http://127.0.0.1:{port}")

    await server.serve_forever()


class Recorder(mp.Process):
    def __init__(self, teleop, dataset, wrist, front):
        super().__init__()

        if not os.path.exists(dataset):
            f = h5py.File(dataset, "w")
            f.create_group("data")
            self.demo_idx = 0
        else:
            f = h5py.File(dataset, "r+")

        data = f["data"]

        if isinstance(data, h5py.Group):
            self.demo_idx = len(data.keys())
        else:
            raise RuntimeError

        self.f = f
        self.data = data

        self.q = mp.Queue(maxsize=-1)

        self.history = []

        wrist_cap = open_camera(int(wrist))
        front_cap = open_camera(int(front))

        self.wrist_thread = wrist_thread = CameraThread(wrist_cap)
        self.front_thread = front_thread = CameraThread(front_cap)

        wrist_thread.start_thread()
        front_thread.start_thread()

        self.teleop: TeleopThread = teleop

    def new(self, pos, quat):
        self.history = []
        self.teleop.init(pos, quat)

    def get_obs(self):
        q = self.teleop.client.read_q()

        wrist_img = self.wrist_thread.latest_frame
        front_img = self.front_thread.latest_frame

        obs = {
            "qpos": q.pos,
            "qvel": q.vel,
            "front_img": front_img,
            "wrist_img": wrist_img,
        }
        return obs

    def add(self, obs, action):
        self.history.append([obs, action])

    def flush(self):
        self.q.put(self.history)

        self.history = []

        self.teleop.reset()

    def run(self):
        while True:
            history = self.q.get()

            grp = self.data.create_group(f"demo_{self.demo_idx}")
            self.demo_idx += 1

            obs = grp.create_group("obs")

            obs_history = [obs for obs, _ in history]
            action_history = [action for _, action in history]

            obs.create_dataset(
                "qpos", data=np.array([obs["qpos"] for obs in obs_history])
            )
            obs.create_dataset(
                "qvel", data=np.array([obs["qvel"] for obs in obs_history])
            )
            obs.create_dataset(
                "front_img",
                data=np.stack([obs["front_img"] for obs in obs_history]),
            )
            obs.create_dataset(
                "wrist_img",
                data=np.stack([obs["wrist_img"] for obs in obs_history]),
            )

            act = grp.create_group("action")

            act.create_dataset(
                "qpos", data=np.array([action["qpos"] for action in action_history])
            )

            self.f.flush()

    def start_process(self):
        self.daemon = True
        self.start()

    def stop_process(self):
        pass


@click.command()
@click.option("--port", default=65432, help="port to run server on")
@click.option("--wrist", type=int, help="port to run server on")
@click.option("--front", type=int, help="port to run server on")
@click.option("--dataset", default="demo.hdf5", help="port to run server on")
def main(port: int, wrist: int, front: int, dataset: str):
    asyncio.run(async_serve(port, wrist, front, dataset))


if __name__ == "__main__":
    main()
