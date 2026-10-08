import os
import threading
import time
from collections import deque
from pathlib import Path
from threading import Lock, Thread
from typing import cast

import av
import numpy as np
import zarr

chunk_size = 128


class VideoWriter:
    def __init__(self, path, fps):
        self.container = av.open(path, mode="w")

        options = {
            "crf": "21",
            "me": "umh",
            "subme": "9",
            "aq-mode": "2",
            "no-fast-pskip": "1",
            "ref": "4",
        }
        options["keyint"] = str(fps * 10)
        options["bframes"] = "8"

        self.stream = self.container.add_stream("h264", rate=fps, options=options)

        self.path = path

    def write(self, img):
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def flush(self):
        for packet in self.stream.encode():
            self.container.mux(packet)

        self.container.close()

    def delete(self):
        self.flush()

        os.remove(self.path)


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
    def __init__(self, path: str, hz: int):
        self.data_dir = Path(path)
        self.root = zarr.group(store=path, overwrite=False)

        self.video_writers: dict[str, VideoWriter] = {}
        self.hz = hz

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
        flat = flatten_dict(nested)

        self.paths = list(flat.keys())
        self.root.update_attributes({"paths": self.paths})

        if set(flat.keys()) != set(self.paths):
            raise RuntimeError

        lengths = set()

        for path, x in flat.items():
            x = np.asarray(x)

            if "img" in path:
                if path not in self.video_writers:
                    parent_dir = self.data_dir / path
                    parent_dir.mkdir(parents=True, exist_ok=True)

                    if "episode_ends" not in self.root:
                        index = 0
                    else:
                        index = self.root.get_array("episode_ends").shape[0]

                    video_path: Path = parent_dir / f"{index}.mp4"

                    self.video_writers[path] = VideoWriter(video_path, self.hz)

                for img in x:
                    self.video_writers[path].write(img)

                continue

            if path not in self.root:
                z = self.root.create_array(
                    name=path,
                    shape=x.shape,
                    dtype=x.dtype,
                    chunks=[chunk_size, *x.shape[1:]],
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
        if self.paths is None:
            return

        if self.end == self.length:
            return

        if "episode_ends" not in self.root:
            z = self.root.create_array(
                name="episode_ends",
                shape=(1,),
                dtype=np.int64,
                chunks=(chunk_size,),
            )
            z[:] = self.length
        else:
            z = self.root.get_array("episode_ends")
            z.append(self.length)

        for path in self.paths:
            if "img" in path:
                self.video_writers[path].flush()
                del self.video_writers[path]

    def trim(self):
        if "paths" not in self.root.attrs:
            return

        for path in cast(list, self.root.attrs["paths"]):
            if "img" in path:
                if path not in self.video_writers:
                    continue

                self.video_writers[path].delete()
                del self.video_writers[path]
                continue

            z = self.root.get_array(path)
            z.resize((int(self.end), *z.shape[1:]))


class RecordThread(Thread):
    _ready = True

    def __init__(self, recorder: ZarrRecorder, bufsize: int = 128):
        super().__init__()
        self.recorder = recorder

        self.q = deque()
        self.lock = Lock()

        self.stop_event = threading.Event()

        self.buffer = []
        self.bufsize = bufsize

        self.keys = set()
        if self.recorder.paths is not None:
            self.keys = set(self.recorder.paths)

    def run(self):
        while not self.stop_event.is_set():
            while self.lock.locked():
                time.sleep(1e-6)

            self._record_loop()

    def _record_loop(self):
        if len(self.q) == 0:
            time.sleep(1e-3)
            return

        msg = self.q.popleft()

        event = msg["event"]

        if event == "record":
            flat = flatten_dict(msg["data"])

            if len(self.keys) == 0:
                self.keys = set(flat.keys())

            if set(flat.keys()) != self.keys:
                raise ValueError

            self.buffer.append(flat)

            if len(self.buffer) >= self.bufsize:
                self.flush()

        elif event == "save":
            if len(self.buffer) > 0:
                self.flush()

            self.recorder.finish()

            self._ready = True

    def flush(self):
        if self.stop_event.is_set():
            return

        stacked = {
            key: np.stack([flat[key] for flat in self.buffer]) for key in self.keys
        }

        self.recorder.add(stacked)

        self.buffer = []

    def record(self, obs, action):
        self._ready = False

        self.q.append({"event": "record", "data": {"obs": obs, "action": action}})

    def save(self):
        self.q.append({"event": "save"})

    def cancel(self):
        with self.lock:
            self.q.clear()
            self.buffer = []

            self.recorder.trim()

        self._ready = True

    def ready(self):
        return self._ready

    def start_thread(self):
        self.daemon = True
        self.start()

    def stop_thread(self):
        self.stop_event.set()

        self.recorder.trim()
