import asyncio
import json
import threading

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation as R
from websockets.asyncio.server import ServerConnection, serve


def quest_to_flange(pos: ArrayLike, quat: ArrayLike) -> NDArray[np.float64]:
    pos = np.asarray(pos, dtype=np.float64)
    quat = np.asarray(quat, dtype=np.float64)

    m = np.eye(4)

    x, y, z = pos
    m[:3, -1] = np.array([-z, -x, y])

    A = np.array(
        [
            [0, -1, 0],
            [-1, 0, 0],
            [0, 0, -1],
        ]
    ).T
    m[:3, :3] = A.T @ R.from_quat(quat).as_matrix() @ A

    return m


class QuestThread(threading.Thread):
    def __init__(self, port: int):
        super().__init__()

        self.port = port
        self.server = None

        self.listeners = {}
        self.async_listeners = {}

        self.anchor = np.eye(4)

    def subscribe(self, name: str):
        def wrapper(fn) -> None:
            if name not in self.listeners:
                self.listeners[name] = [fn]
            else:
                self.listeners[name].append(fn)

        return wrapper

    def async_subscribe(self, name: str):
        def wrapper(fn) -> None:
            if name not in self.listeners:
                self.async_listeners[name] = [fn]
            else:
                self.async_listeners[name].append(fn)

        return wrapper

    async def handler(self, websocket: ServerConnection):
        async for message in websocket:
            msg = json.loads(message)

            event = msg["type"]

            async with asyncio.TaskGroup() as tg:
                for fn in self.async_listeners.get(event, []):
                    tg.create_task(fn(websocket, msg))

                for fn in self.listeners.get(event, []):
                    fn(msg)

    async def async_serve(self):
        self.server = await serve(self.handler, host="0.0.0.0", port=self.port)

        await self.server.serve_forever()

    def run(self):
        asyncio.run(self.async_serve())

    def start_thread(self):
        self.daemon = True
        self.start()

    def stop_thread(self):
        if self.server is not None:
            self.server.close()

    def set_anchor(self, frame: ArrayLike):
        self.anchor = np.asarray(frame, dtype=np.float64)
