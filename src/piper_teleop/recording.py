import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Literal, cast

import numpy as np
import optree
import zarr
from imagecodecs.zarr import ArrayBytesCodec, Jpeg, register_codecs
from numpy.typing import NDArray
from optree.pytree import PyTreeSpec

register_codecs()


def makedirs(path: str | os.PathLike):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def array_keys(root: zarr.Group) -> set[str]:
    array_keys = list(root.array_keys())
    group_keys = list(root.group_keys())

    while len(group_keys) > 0:
        new_keys = []
        for name in group_keys:
            g = root.get_group(name)

            array_keys.extend([f"{name}/{key}" for key in g.array_keys()])
            new_keys.extend([f"{name}/{key}" for key in g.group_keys()])

        group_keys = new_keys

    return set(array_keys)


def treespec_names(treespec: PyTreeSpec) -> list[tuple[str, ...]]:
    paths = optree.treespec_paths(treespec)
    names = [tuple(str(name) for name in path) for path in paths]
    return names


def group_tree(group: zarr.Group):
    tree = {}

    for key in group.array_keys():
        tree[key] = group.get_array(key)

    for key in group.group_keys():
        tree[key] = group_structure(group.get_group(key))

    return tree


def group_structure(group: zarr.Group):
    treespec = {}

    for key in group.array_keys():
        treespec[key] = optree.treespec.leaf()

    for key in group.group_keys():
        treespec[key] = group_structure(group.get_group(key))

    if len(treespec.values()) > 0:
        return optree.treespec.dict(treespec)

    return None


@dataclass(frozen=True)
class Image:
    frame: NDArray[np.uint8]
    codec: ArrayBytesCodec | Literal["auto"] = "auto"

    def __post_init__(self):
        if self.codec == "auto":
            object.__setattr__(self, "codec", Jpeg(level=90))


class DatasetWriter:
    def __init__(
        self, path: str | os.PathLike, *, shard_size: int = 32, buffer_size: int = 128
    ):
        root_dir = Path(path)

        makedirs(root_dir.parent)

        self._group = zarr.group(path)
        self._treespec = group_structure(self._group)

        self.shard_size = shard_size

        self._buffer = []
        self.buffer_size = buffer_size

        if self._treespec is None:
            self._group.update_attributes({"length": 0, "episode_ends": []})

        self._length = cast(int, self._group.attrs["length"])

    def add(self, tree: Any, /) -> None:
        leaves, treespec = optree.tree_flatten(tree)

        if self._treespec is None:
            self._treespec = treespec

        elif treespec != self._treespec:
            raise ValueError

        self._buffer.append(leaves)

        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def mark_last(self):
        self.flush()

        self._group.update_attributes({"length": self._length})

        ends = cast(list[int], self._group.attrs["episode_ends"])
        ends.append(self._length)

        self._group.update_attributes({"episode_ends": ends})

    def flush(self) -> None:
        if len(self._buffer) == 0:
            return

        names = treespec_names(self._treespec)

        for i, name in enumerate(names):
            name = os.path.join(*name)

            if isinstance(self._buffer[0][i], Image):
                x = np.stack([buf[i].frame for buf in self._buffer])

                serializer = self._buffer[0][i].codec
                chunk_size = 1
            else:
                x = np.stack([buf[i] for buf in self._buffer])

                serializer = "auto"
                chunk_size = self.shard_size

            length = x.shape[0]

            if name in self._group:
                z = self._group.get_array(name)
                z.resize((z.shape[0] + length, *z.shape[1:]))

            else:
                z = self._group.create_array(
                    name,
                    shape=x.shape,
                    dtype=x.dtype,
                    chunks=[chunk_size, *x.shape[1:]],
                    shards=[self.shard_size, *x.shape[1:]],
                    compressors=None,
                    serializer=serializer,
                )

            z[-length:] = x

        self._length += length
        self._buffer = []

    def prune(self) -> None:
        length = cast(int, self._group.attrs["length"])

        if self._treespec is None:
            return

        names = treespec_names(self._treespec)

        for name in names:
            name = os.path.join(*name)

            z = self._group.get_array(name)

            if z.shape[0] > length:
                z.resize([length, *z.shape[1:]])


class RecordThread(Thread):
    _ready = True

    def __init__(self, writer: DatasetWriter):
        super().__init__()
        self.writer = writer

        self.q = deque()
        self.lock = Lock()

        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            with self.lock:
                self._record_loop()

    def _record_loop(self):
        if len(self.q) == 0:
            time.sleep(1e-3)
            return

        msg = self.q.popleft()

        event = msg["event"]

        if event == "record":
            self.writer.add(msg["data"])

        elif event == "save":
            self.writer.mark_last()

            self._ready = True

    def record(self, obs, action):
        self._ready = False

        self.q.append({"event": "record", "data": {"obs": obs, "action": action}})

    def save(self):
        self.q.append({"event": "save"})

    def cancel(self):
        with self.lock:
            self.q.clear()

            self.writer.flush()
            self.writer.prune()

            self._ready = True

    def ready(self):
        return self._ready

    def start_thread(self):
        self.daemon = True
        self.start()

    def stop_thread(self):
        self.stop_event.set()

        self.writer.flush()
        self.writer.prune()
