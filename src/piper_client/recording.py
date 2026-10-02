import threading
import time
from queue import Queue
from typing import cast

import numpy as np
import zarr
from numpy.typing import ArrayLike, NDArray
from zarr.codecs import BloscCodec
from zarr.core.array_spec import ArrayConfigLike
from zarr.storage import LocalStore


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


class ZarrRecorder:
    def __init__(self, path: str):
        self.root = zarr.group(store=path, overwrite=False)

        self.length = self.end

        self.paths = None
        if "paths" in self.root.attrs:
            self.paths = cast(list, self.root.attrs["paths"])

    @property
    def end(self):
        end = np.array([0], dtype=np.int64)

        if "episode_ends" in self.root:
            z = self.root.get_array("episode_ends")

            if z.shape[0] > 0:
                i = z.shape[0]
                end = np.asarray(z[i - 1 : i], dtype=np.int64)

        return end

    def add(self, nested: dict):
        compressors = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")

        flat = flatten_dict(nested)

        self.paths = list(flat.keys())
        self.root.update_attributes({"paths": self.paths})

        if set(flat.keys()) != set(self.paths):
            raise RuntimeError

        lengths = set()

        for path, x in flat.items():
            x = np.asarray(x)

            if path not in self.root:
                z = self.root.create_array(
                    name=path,
                    shape=x.shape,
                    dtype=x.dtype,
                    chunks=[1, *x.shape[1:]],
                    compressors=compressors,
                )
                z[:] = x
            else:
                z = self.root.get_array(path)
                z.append(x)

            lengths.add(x.shape[0])

        if len(lengths) != 1:
            raise RuntimeError

        self.length += int(lengths.pop())

    def finish(self):
        if self.end == self.length:
            return

        if "episode_ends" not in self.root:
            z = self.root.create_array(
                name="episode_ends",
                shape=(1,),
                dtype=np.int64,
                chunks=(1,),
            )
            z[:] = self.length
        else:
            z = self.root.get_array("episode_ends")
            z.append(self.length)

    def close(self):
        for path in cast(list, self.root.attrs["paths"]):
            z = self.root.get_array(path)
            z.resize((int(self.end), *z.shape[1:]))


class RecordThread(threading.Thread):
    def __init__(self, recorder: ZarrRecorder, bufsize: int = 128):
        super().__init__()
        self.recorder = recorder

        self.q = Queue(maxsize=-1)

        self.stop_event = threading.Event()

        self.buffer: list[dict[str, dict[str, ArrayLike]]] = []
        self.bufsize = bufsize

    def run(self):
        while not self.stop_event.is_set():
            if self.q.empty():
                time.sleep(1e-3)

            val = self.q.get()

            if val is None:
                if len(self.buffer) > 0:
                    self.flush()

                self.recorder.finish()
            else:
                obs, action = val
                self.buffer.append({"obs": obs, "action": action})

                if len(self.buffer) >= self.bufsize:
                    self.flush()

    def flush(self):
        flat_buffer = [flatten_dict(buf) for buf in self.buffer]

        paths = flat_buffer[0].keys()

        stacked = {}
        for path in paths:
            stacked[path] = np.stack([buf[path] for buf in flat_buffer])

        self.recorder.add(stacked)

        self.buffer = []

    def record(self, obs, action):
        self.q.put((obs, action))

    def save(self):
        self.q.put(None)

    def start_thread(self):
        self.daemon = True
        self.start()

    def stop_thread(self):
        while not self.q.empty():
            time.sleep(1)

        self.stop_event.set()

        if len(self.buffer) > 0:
            self.flush()

        self.recorder.close()
