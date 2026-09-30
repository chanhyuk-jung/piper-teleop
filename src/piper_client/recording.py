import threading
from queue import Queue
from typing import Any

import numpy as np
import zarr
from zarr.codecs import BloscCodec


class ZarrRecorder:
    def __init__(self, path: str):
        self.root = zarr.group(path, overwrite=False)

        if "episode_ends" not in self.root:
            self.ends = self.root.create_array(
                name="episode_ends", shape=(0,), chunks=(1,), dtype="int64"
            )

            self.obs = self.root.create_group("obs")
            self.action = self.root.create_group("action")
        else:
            self.ends = self.root.get_array("episode_ends")

            self.obs = self.root.get_group("obs")
            self.action = self.root.get_group("action")

        if self.ends.shape[0] == 0:
            self.end = np.array([0], dtype=np.int64)
        else:
            self.end = np.asarray([self.ends[-1]], dtype=np.int64)

    def add(self, obs: dict[str, Any], action: dict[str, Any]):
        for name, x in action.items():
            x = np.asarray(x)[None, :]

            if name not in self.action:
                compressors = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")

                self.action.create_array(
                    name=name,
                    shape=x.shape,
                    dtype=x.dtype,
                    chunks=[1, *x.shape[1:]],
                    compressors=compressors,
                )
            else:
                z = self.action.get_array(name)
                z.append(x)

        for name, x in obs.items():
            x = np.asarray(x)[None, :]

            if name not in self.obs:
                compressors = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")

                self.obs.create_array(
                    name=name,
                    shape=x.shape,
                    dtype=x.dtype,
                    chunks=[1, *x.shape[1:]],
                    compressors=compressors,
                )
            else:
                z = self.obs.get_array(name)
                z.append(x)

        self.end += 1

    def finish(self):
        if self.ends.shape[0] == 0:
            end = 0
        else:
            end = self.ends[-1]

        if self.end == 0 or self.end == end:
            return

        self.ends.append(self.end)


class RecordThread(threading.Thread):
    def __init__(self, recorder: ZarrRecorder):
        super().__init__()
        self.recorder = recorder

        self.q = Queue(maxsize=-1)

        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            val = self.q.get()

            if val is None:
                self.recorder.finish()
            else:
                obs, action = val

                self.recorder.add(obs, action)

    def record(self, obs, action):
        self.q.put((obs, action))

    def save(self):
        self.q.put(None)

    def start_thread(self):
        self.start()

    def stop_thread(self):
        self.stop_event.set()
