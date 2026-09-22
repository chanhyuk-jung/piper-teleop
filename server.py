import asyncio
import json
import os

import click
import h5py
import numpy as np
from websockets.asyncio.server import ServerConnection, serve

from camera import CameraThread, open_camera
from piper_utils import RealPiper


async def async_serve(port: int, wrist_cam, front_cam, dataset):
    piper = RealPiper("can0", urdf_path="piper")
    piper.start()

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
            demo_idx = len(f["data"].keys())

        history = []

        async for message in websocket:
            msg = json.loads(message)
            payload = msg["payload"]

            if msg["type"] == "start":
                piper.init(payload["position"], payload["quaternion"])

                history = []

            elif msg["type"] == "move":
                q = piper.read_q()

                front_img = front_thread.async_read()
                wrist_img = wrist_thread.async_read()

                obs = {
                    "qpos": q.pos,
                    "qvel": q.vel,
                    "front_img": front_img,
                    "wirst_img": wrist_img,
                }

                piper.update_target(
                    payload["position"],
                    payload["quaternion"],
                    payload["gripper"],
                    msg["delta"],
                )

                qpos = piper.k.get_qpos()

                action = {"qpos": qpos}

                history.append([obs, action])

            elif msg["type"] == "stop":
                piper.clear_tqrget_q()

            elif msg["type"] == "reset":
                piper.clear_tqrget_q()

                piper.move_qpos([0] * 7, timeout=10)
                piper.k.set_qpos([0] * 7)

                grp = f["data"].create_group(f"demo_{demo_idx}")
                demo_idx += 1

                grp.create_group("obs")

                obs_history = [obs for obs, _ in history]
                action_history = [action for _, action in history]

                grp["obs"].create_dataset(
                    "qpos", data=np.array([obs["qpos"] for obs in obs_history])
                )
                grp["obs"].create_dataset(
                    "qvel", data=np.array([obs["qvel"] for obs in obs_history])
                )
                grp["obs"].create_dataset(
                    "front_img",
                    data=np.stack([obs["front_img"] for obs in obs_history]),
                )
                grp["obs"].create_dataset(
                    "wrist_img",
                    data=np.stack([obs["wrist_img"] for obs in obs_history]),
                )

                grp.create_group("action")

                grp["action"].create_dataset(
                    "qpos", data=np.array([action["qpos"] for action in action_history])
                )

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
