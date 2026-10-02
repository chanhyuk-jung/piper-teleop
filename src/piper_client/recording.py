import threading
import time
from queue import Queue

import numpy as np
import zarr
from numpy.typing import ArrayLike
from zarr.codecs import BloscCodec
from zarr.storage import LocalStore


class ZarrRecorder:
    def __init__(self, path: str):
        self.store = LocalStore(path)
        self.root = zarr.group(store=self.store, overwrite=False)

        if "episode_ends" not in self.root:
            self.root.create_group("obs")
            self.root.create_group("action")

            self.root.create_array(
                name="episode_ends", shape=(0,), chunks=(1,), dtype="int64"
            )

        self.obs = self.root.get_group("obs")
        self.action = self.root.get_group("action")
        self.ends = self.root.get_array("episode_ends")

        self.end = np.array([0], dtype=np.int64)

        if self.ends.shape[0] > 0:
            self.end = np.asarray([self.ends[-1]], dtype=np.int64)

        self.trim(int(self.end))

    def add(
        self,
        obs: dict[str, ArrayLike | dict[str, ArrayLike]],
        action: dict[str, ArrayLike | dict[str, ArrayLike]],
    ):
        for name, x in action.items():
            x = np.asarray(x)
            x = x[None, ...]

            if name not in self.action.array_keys():
                compressors = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")

                z = self.action.create_array(
                    name=name,
                    shape=x.shape,
                    dtype=x.dtype,
                    chunks=x.shape,
                    compressors=compressors,
                )
                z[:] = x
            else:
                z = self.action.get_array(name)
                z.append(x)

        for name, x in obs.items():
            x = np.asarray(x)
            x = x[None, ...]

            if name not in self.obs.array_keys():
                compressors = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")

                z = self.obs.create_array(
                    name=name,
                    shape=x.shape,
                    dtype=x.dtype,
                    chunks=x.shape,
                    compressors=compressors,
                )
                z[:] = x
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

    def check(self):
        length = 0
        if self.ends.shape[0] > 0:
            length = self.ends[-1]

        for name in self.action:
            if length != self.action.get_array(name).shape[0]:
                return False

        for name in self.obs:
            if length != self.obs.get_array(name).shape[0]:
                return False

        return True

    def trim(self, length: int):
        for name in self.action:
            x = self.action.get_array(name)
            x.resize([length, *x.shape[1:]])

        for name in self.obs:
            x = self.obs.get_array(name)
            x.resize([length, *x.shape[1:]])

    def close(self):
        self.store.close()


class RecordThread(threading.Thread):
    def __init__(self, recorder: ZarrRecorder):
        super().__init__()
        self.recorder = recorder

        self.q = Queue(maxsize=-1)

        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            if self.q.empty():
                continue

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
        while not self.q.empty():
            time.sleep(1)

        self.stop_event.set()
        self.recorder.close()
