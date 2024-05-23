import argparse
import json
import time
from threading import RLock, Thread
from typing import List, Optional

import cv2


class Camera:
    def __init__(self, path, resolution: List[int] = [1280, 720], fps=30.0) -> None:
        self.path = path

        self.cap = cv2.VideoCapture()
        self.fps = fps
        self.resolution = resolution
        self.set_cap(self.resolution, self.fps)
        print(f"Init camera: {self.path} with {self.resolution} ({self.fps})")

    def set_cap(self, resolution, fps):
        """Setup camera
        - Args
            - resolution: A tuple with width and height, e.g.( <width>, <height> )
            - fps: The fps
        """
        if isinstance(self.path, str) and self.path.isdigit():
            self.path = int(self.path)
        status = self.cap.open(self.path, cv2.CAP_V4L)
        if not status:
            raise RuntimeError(f"Can not open the camera from {self.path}")

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 4.0)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        if resolution:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

            cur_resol = (
                self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            )
            self.resolution = cur_resol

        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)

    def reset_cap(self):
        self.set_cap(self.resolution, self.fps)
        print("Reset Camera")
        time.sleep(1 / 30)

    def read(self):
        status, frame = self.cap.read()

        if not status:
            self.reset_cap()
            status, frame = self.cap.read()

            if not status:
                raise RuntimeError(
                    "Can not capture frame, please make sure the camera is available"
                )

        return frame

    def get_fps(self):
        return self.fps

    def alive(self):
        return self.cap.isOpened()

    def release(self):
        self.cap.release()


class CameraStream(Camera):
    def __init__(
        self,
        path,
        resolution: List[int] = [1280, 720],
        fps=30.0,
        start_loop: bool = True,
    ) -> None:
        super().__init__(path, resolution, fps)
        self.t = Thread(target=self._update_event, daemon=True)
        self.lock = RLock()
        self.is_stop = False
        self._frame = None
        if start_loop:
            self.start()

    def start(self):
        if self.t.is_alive():
            return
        self.t.start()
        while self._frame is None:
            time.sleep(0.3)
            continue

    def _update_event(self):
        while not self.is_stop:
            ret, frame = self.cap.read()
            if not ret:
                break
            with self.lock:
                self._frame = frame

        self.cap.release()
        print("Stop Streaming")

    @property
    def frame(self):
        with self.lock:
            return self._frame

    def release(self):
        self.is_stop = True
        if self.t.is_alive():
            self.t.join()


def valid_camera_parser(args: argparse.Namespace) -> argparse.Namespace:
    args.device = int(args.device) if args.device.isdigit() else args.device
    args.fps = int(args.fps) if args.fps.isdigit() else args.fps
    args.resolution = json.loads(args.resolution)
    return args


def camera_parser(
    parser: Optional[argparse.ArgumentParser] = None, group_nam: str = "Camera"
):
    if parser is None:
        parser = argparse.ArgumentParser()
    group = parser.add_argument_group(group_nam)
    group.add_argument("-d", "--device", type=str, default="0")
    group.add_argument(
        "-r", "--resolution", type=str, default="[1920, 1080]", help="[width, height]"
    )
    group.add_argument("-f", "--fps", type=str, default="30")
    return parser


def test_cam():
    args = valid_camera_parser(camera_parser().parse_args())
    print(f"Input: {args}")

    cam = Camera(path=args.device, resolution=args.resolution, fps=args.fps)
    while cam.alive():
        frame = cam.read()
        cv2.imshow("test", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


def test_cam_stream():
    cam = CameraStream("/dev/video0")

    while cam.alive():
        cv2.imshow("test", cam.frame)
        if cv2.waitKey(1) in [ord("q"), 27]:
            break
    cam.release()


if __name__ == "__main__":
    # test_cam()
    test_cam_stream()
