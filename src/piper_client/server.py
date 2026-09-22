import asyncio
import json
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

    wrist_cap = open_camera(int(wrist_cam))
    front_cap = open_camera(int(front_cam))

    wrist_thread = CameraThread(wrist_cap)
    front_thread = CameraThread(front_cap)

    wrist_thread.start_thread()
    front_thread.start_thread()

    dataset = "demo.hdf5"

    async def handler(websocket: ServerConnection):
        if not os.path.exists(dataset):
            f = h5py.File(dataset, "w")
            f.create_group("data")
            demo_idx = 0
        else:
            f = h5py.File(dataset, "r+")

        data = f["data"]

        if isinstance(data, h5py.Group):
            demo_idx = len(data.keys())
        else:
            raise RuntimeError

        history = []

        async for message in websocket:
            msg = json.loads(message)
            payload = msg["payload"]

            if msg["type"] == "start":
                teleop.init(payload["position"], payload["quaternion"])

                obs_history = []
                action_history = []

            elif msg["type"] == "move":
                q = teleop.client.read_q()

                front_img = front_thread.latest_frame
                wrist_img = wrist_thread.latest_frame

                teleop.update_target(
                    payload["position"],
                    payload["quaternion"],
                    payload["gripper"],
                    msg["delta"],
                )

                action = {"qpos": teleop.action}

                obs = {
                    "qpos": q.pos,
                    "qvel": q.vel,
                    "front_img": front_img,
                    "wrist_img": wrist_img,
                }

                history.append([obs, action])

            elif msg["type"] == "stop":
                teleop.pause()

                grp = data.create_group(f"demo_{demo_idx}")
                demo_idx += 1

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

                f.flush()

            elif msg["type"] == "reset":
                teleop.reset()

    server = await serve(handler, host="0.0.0.0", port=port)
    print(f"websocket server running at http://127.0.0.1:{port}")

    await server.serve_forever()


@click.command()
@click.option("--port", default=65432, help="port to run server on")
@click.option("--wrist", type=int, help="port to run server on")
@click.option("--front", type=int, help="port to run server on")
@click.option("--dataset", default="demo.hdf5", help="port to run server on")
def main(port: int, wrist: int, front: int, dataset: str):
    asyncio.run(async_serve(port, wrist, front, dataset))


if __name__ == "__main__":
    main()
