#!/usr/bin/python3
import asyncio
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List

import cv2

from async_yolo import camera, yolo
from async_yolo.visualization import BBoxVisualization, Displayer, show_fps
from async_yolo.yolo_classes import get_cls_dict


@dataclass
class Args:
    engine: str = "/home/nvidia/workspace/jetson-orin-multicam/yolo/yolov4-tiny-416.trt"
    cam0: str = "/dev/video0"
    cam1: str = "/dev/video1"
    category_num: int = 80
    threshold: float = 0.3


async def async_main(loop: asyncio.AbstractEventLoop, exec: ThreadPoolExecutor):
    args = Args()
    inference_fps, display_fps = 15, 30 * 3
    vis = BBoxVisualization(get_cls_dict(args.category_num))
    cams: List[camera.CameraStream] = await asyncio.gather(
        loop.run_in_executor(exec, camera.CameraStream, "/dev/video0"),
        loop.run_in_executor(exec, camera.CameraStream, "/dev/video1"),
    )
    inferences: List[yolo.StreamYOLO] = await asyncio.gather(
        loop.run_in_executor(
            exec,
            yolo.StreamYOLO,
            "/home/nvidia/workspace/jetson-orin-multicam/yolo/yolov4-tiny-416.trt",
            args.threshold,
            inference_fps,
        ),
        loop.run_in_executor(
            exec,
            yolo.StreamYOLO,
            "/home/nvidia/workspace/jetson-orin-multicam/yolo/yolov4-tiny-416.trt",
            args.threshold,
            inference_fps,
        ),
    )
    t_display = time.time()
    t_infer = time.time()
    fps = 0.0
    tic = time.time()

    def can_display():
        return time.time() - t_display >= (1 / display_fps)

    def can_set_input(t_infer: float):
        return time.time() - t_infer > 1 / inference_fps

    def block_infer(
        cam: camera.Camera,
        inference: yolo.StreamYOLO,
        vis: BBoxVisualization,
    ):
        nonlocal t_infer
        frame = copy.deepcopy(cam.frame)
        if can_set_input(t_infer):
            inference.set_input(frame)
            t_infer = time.time()
        return vis.draw_bboxes(frame, *inference.output)

    displayer = Displayer()
    frames, draws = (), ()
    while True:
        if can_display():
            draws = await asyncio.gather(
                loop.run_in_executor(exec, block_infer, cams[0], inferences[0], vis),
                loop.run_in_executor(exec, block_infer, cams[1], inferences[1], vis),
            )
            concat_frame = show_fps(cv2.hconcat(draws), fps)

            if not displayer.show(concat_frame):
                break

            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            fps = curr_fps if fps == 0.0 else (fps * 0.95 + curr_fps * 0.05)
            tic = toc
            t_display = time.time()

    del frames
    del draws
    await asyncio.gather(
        loop.run_in_executor(exec, displayer.release),
        loop.run_in_executor(exec, cams[0].release),
        loop.run_in_executor(exec, cams[1].release),
        loop.run_in_executor(exec, inferences[0].release),
        loop.run_in_executor(exec, inferences[1].release),
    )


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=2)
    loop.run_until_complete(async_main(loop, executor))
