import argparse
import asyncio
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import cv2

from async_yolo import camera, yolo
from async_yolo.visualization import BBoxVisualization, Displayer, show_fps
from async_yolo.yolo_classes import get_cls_dict


def infer_parser(parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:
    """Parse input arguments."""
    desc = (
        "Capture and display live camera video, while doing "
        "real-time object detection with TensorRT optimized "
        "YOLO model on Jetson"
    )

    if parser is None:
        parser = argparse.ArgumentParser(description=desc)

    group = parser.add_argument_group("Inference")
    group.add_argument("-m", "--model", type=str, required=True, help=("path to model"))
    group.add_argument(
        "-c",
        "--category_num",
        type=int,
        default=80,
        help="number of object categories [80]",
    )
    group.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.3,
        help="set the detection confidence threshold",
    )
    group.add_argument(
        "-l",
        "--letter_box",
        action="store_true",
        help="inference with letterboxed image [False]",
    )
    return parser


def valid_infer_parser(args: argparse.Namespace) -> argparse.Namespace:
    if args.category_num <= 0:
        raise SystemExit(f"ERROR: bad category_num ({args.category_num})!")
    if not Path(args.model).exists():
        raise SystemExit(f"ERROR: can not find model {args.model}")
    return args


def main():
    args = valid_infer_parser(infer_parser().parse_args())
    inference_fps, display_fps = 15, 30 * 2 + 15
    vis = BBoxVisualization(get_cls_dict(args.category_num))
    cams: List[camera.CameraStream] = [
        camera.CameraStream("/dev/video0"),
        camera.CameraStream("/dev/video1"),
    ]
    inferences: List[yolo.StreamYOLO] = [
        yolo.StreamYOLO(
            model_path=args.model,
            threshold=args.threshold,
            limit_inference_fps=inference_fps,
        ),
        yolo.StreamYOLO(
            model_path=args.model,
            threshold=args.threshold,
            limit_inference_fps=inference_fps,
        ),
    ]
    t_display = time.time()
    t_infer = time.time()
    fps = 0.0
    tic = time.time()

    def can_display():
        return time.time() - t_display >= (1 / display_fps)

    def can_set_input():
        return time.time() - t_infer > 1 / inference_fps

    displayer = Displayer()
    while True:
        if can_display():
            frame0 = copy.deepcopy(cams[0].frame)
            frame1 = copy.deepcopy(cams[1].frame)
            if can_set_input():
                inferences[0].set_input(frame0)
                inferences[1].set_input(frame1)
                t_infer = time.time()

            frame0 = vis.draw_bboxes(frame0, *inferences[0].output)
            frame1 = vis.draw_bboxes(frame1, *inferences[1].output)
            concat_frame = show_fps(cv2.vconcat([frame0, frame1]), fps)

            if not displayer.show(concat_frame):
                break

            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            fps = curr_fps if fps == 0.0 else (fps * 0.95 + curr_fps * 0.05)
            tic = toc
            t_display = time.time()

    [cam.release() for cam in cams]
    [inference.release() for inference in inferences]
    displayer.release()


async def async_main(loop: asyncio.AbstractEventLoop, exec: ThreadPoolExecutor):
    args = valid_infer_parser(infer_parser().parse_args())
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
            args.model,
            args.threshold,
            inference_fps,
        ),
        loop.run_in_executor(
            exec,
            yolo.StreamYOLO,
            args.model,
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
            concat_frame = show_fps(cv2.vconcat(draws), fps)

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
