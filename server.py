import asyncio
import json
import queue
import threading

import click
from scipy.spatial.transform import Rotation as R
from websockets.asyncio.server import ServerConnection, serve

from piper_utils import Piper


async def async_serve(port: int):
    piper = Piper("can0", urdf_path="piper")
    piper.start()

    async def handler(websocket: ServerConnection):
        async for message in websocket:
            msg = json.loads(message)
            payload = msg["payload"]

            if msg["type"] == "start":
                piper.init(payload["position"], payload["quaternion"])

            elif msg["type"] == "move":
                piper.update_target(
                    payload["position"],
                    payload["quaternion"],
                    payload["gripper"],
                    msg["delta"],
                )

            elif msg["type"] == "stop":
                piper.clear_tqrget_q()

            elif msg["type"] == "reset":
                piper.clear_tqrget_q()

                piper.move_qpos([0] * 7)
                piper.k.set_qpos([0] * 7)

    server = await serve(handler, host="0.0.0.0", port=port)
    print(f"websocket server running at http://127.0.0.1:{port}")

    await server.serve_forever()


@click.command()
@click.option("--port", default=65432, help="port to run server on")
def main(port: int):
    asyncio.run(async_serve(port))


if __name__ == "__main__":
    main()
