import math
import time
from threading import Event, Lock, Thread

import cv2


def open_camera(index, fps=30, width=640, height=480, warmup_s=1):
    cap = cv2.VideoCapture(index)
    success = cap.set(cv2.CAP_PROP_FPS, float(fps))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    if not success or not math.isclose(fps, actual_fps, rel_tol=1e-3):
        raise RuntimeError(f"failed to set fps={fps} ({actual_fps=}).")

    width_success = cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    height_success = cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(height))

    actual_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    if not width_success or width != actual_width:
        raise RuntimeError(
            f"failed to set {width=} ({actual_width=}, {width_success=})."
        )

    actual_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if not height_success or height != actual_height:
        raise RuntimeError(
            f"failed to set {height=} ({actual_height=}, {height_success=})."
        )

    start_time = time.time()
    while time.time() - start_time < warmup_s:
        cap.read()
        time.sleep(0.1)

    return cap


class CameraThread(Thread):
    def __init__(self, cap):
        super().__init__()

        self.stop_event = Event()
        self.frame_lock = Lock()
        self.new_frame_event = Event()

        self.cap = cap

        self.latest_frame = None

    def run(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()

            if not ret or frame is None:
                raise RuntimeError(f"read failed (status={ret}).")

            with self.frame_lock:
                self.latest_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.new_frame_event.set()

    def async_read(self):
        if not self.new_frame_event.wait(timeout=1):
            raise TimeoutError

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        return frame

    def start_thread(self):
        self.daemon = True
        self.start()

    def stop_thread(self):
        self.stop_event.set()
