import ctypes
import os
import time
from contextlib import contextmanager
from pathlib import Path
from threading import RLock, Thread

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

# ======================================================


def init_cuda():
    from pycuda import autoinit  # type: ignore  # noqa: I001, F401


def load_yolo_plugin(path: str = "./plugins/libyolo_layer.so"):
    try:
        ctypes.cdll.LoadLibrary(path)
    except OSError as e:
        raise SystemExit(
            f"ERROR: failed to load {path}.  "
            'Did you forget to do a "make" in the "./plugins/" '
            "subdirectory?"
        ) from e


# ======================================================


def _preprocess_yolo(img, input_shape, letter_box=False):
    """Preprocess an image before TRT YOLO inferencing.

    # Args
        img: int8 numpy array of shape (img_h, img_w, 3)
        input_shape: a tuple of (H, W)
        letter_box: boolean, specifies whether to keep aspect ratio and
                    create a "letterboxed" image for inference

    # Returns
        preprocessed img: float32 numpy array of shape (3, H, W)
    """
    if letter_box:
        img_h, img_w, _ = img.shape
        new_h, new_w = input_shape[0], input_shape[1]
        offset_h, offset_w = 0, 0
        if (new_w / img_w) <= (new_h / img_h):
            new_h = int(img_h * new_w / img_w)
            offset_h = (input_shape[0] - new_h) // 2
        else:
            new_w = int(img_w * new_h / img_h)
            offset_w = (input_shape[1] - new_w) // 2
        resized = cv2.resize(img, (new_w, new_h))
        img = np.full((input_shape[0], input_shape[1], 3), 127, dtype=np.uint8)
        img[offset_h : (offset_h + new_h), offset_w : (offset_w + new_w), :] = resized
    else:
        img = cv2.resize(img, (input_shape[1], input_shape[0]))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    return img


def _nms_boxes(detections, nms_threshold):
    """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding
    boxes with their confidence scores and return an array with the
    indexes of the bounding boxes we want to keep.

    # Args
        detections: Nx7 numpy arrays of
                    [[x, y, w, h, box_confidence, class_id, class_prob],
                     ......]
    """
    x_coord = detections[:, 0]
    y_coord = detections[:, 1]
    width = detections[:, 2]
    height = detections[:, 3]
    box_confidences = detections[:, 4] * detections[:, 6]

    areas = width * height
    ordered = box_confidences.argsort()[::-1]

    keep = list()
    while ordered.size > 0:
        # Index of the current element:
        i = ordered[0]
        keep.append(i)
        xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
        yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
        xx2 = np.minimum(
            x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]]
        )
        yy2 = np.minimum(
            y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]]
        )

        width1 = np.maximum(0.0, xx2 - xx1 + 1)
        height1 = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width1 * height1
        union = areas[i] + areas[ordered[1:]] - intersection
        iou = intersection / union
        indexes = np.where(iou <= nms_threshold)[0]
        ordered = ordered[indexes + 1]

    keep = np.array(keep)
    return keep


def _postprocess_yolo(
    trt_outputs, img_w, img_h, conf_th, nms_threshold, input_shape, letter_box=False
):
    """Postprocess TensorRT outputs.

    # Args
        trt_outputs: a list of 2 or 3 tensors, where each tensor
                    contains a multiple of 7 float32 numbers in
                    the order of [x, y, w, h, box_confidence, class_id, class_prob]
        conf_th: confidence threshold
        letter_box: boolean, referring to _preprocess_yolo()

    # Returns
        boxes, scores, classes (after NMS)
    """
    # filter low-conf detections and concatenate results of all yolo layers
    detections = []
    for o in trt_outputs:
        dets = o.reshape((-1, 7))
        dets = dets[dets[:, 4] * dets[:, 6] >= conf_th]
        detections.append(dets)
    detections = np.concatenate(detections, axis=0)

    if len(detections) == 0:
        boxes = np.zeros((0, 4), dtype=np.int)
        scores = np.zeros((0,), dtype=np.float32)
        classes = np.zeros((0,), dtype=np.float32)
    else:
        box_scores = detections[:, 4] * detections[:, 6]

        # scale x, y, w, h from [0, 1] to pixel values
        old_h, old_w = img_h, img_w
        offset_h, offset_w = 0, 0
        if letter_box:
            if (img_w / input_shape[1]) >= (img_h / input_shape[0]):
                old_h = int(input_shape[0] * img_w / input_shape[1])
                offset_h = (old_h - img_h) // 2
            else:
                old_w = int(input_shape[1] * img_h / input_shape[0])
                offset_w = (old_w - img_w) // 2
        detections[:, 0:4] *= np.array([old_w, old_h, old_w, old_h], dtype=np.float32)

        # NMS
        nms_detections = np.zeros((0, 7), dtype=detections.dtype)
        for class_id in set(detections[:, 5]):
            idxs = np.where(detections[:, 5] == class_id)
            cls_detections = detections[idxs]
            keep = _nms_boxes(cls_detections, nms_threshold)
            nms_detections = np.concatenate(
                [nms_detections, cls_detections[keep]], axis=0
            )

        xx = nms_detections[:, 0].reshape(-1, 1)
        yy = nms_detections[:, 1].reshape(-1, 1)
        if letter_box:
            xx = xx - offset_w
            yy = yy - offset_h
        ww = nms_detections[:, 2].reshape(-1, 1)
        hh = nms_detections[:, 3].reshape(-1, 1)
        boxes = np.concatenate([xx, yy, xx + ww, yy + hh], axis=1) + 0.5
        boxes = boxes.astype(np.int)
        scores = nms_detections[:, 4] * nms_detections[:, 6]
        classes = nms_detections[:, 5]
    return boxes, scores, classes


# ======================================================


class HostDeviceMem:
    """Simple helper data class that's a little nicer to use than a 2-tuple."""

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        del self.device
        del self.host


def get_input_shape(engine):
    """Get input shape of the TensorRT YOLO engine."""
    binding = engine[0]
    assert engine.binding_is_input(binding)
    binding_dims = engine.get_binding_shape(binding)
    if len(binding_dims) == 4:
        return tuple(binding_dims[2:])
    elif len(binding_dims) == 3:
        return tuple(binding_dims[1:])
    else:
        raise ValueError("bad dims of binding %s: %s" % (binding, str(binding_dims)))


def allocate_buffers(engine):
    """Allocates all host/device in/out buffers required for an engine."""
    inputs = []
    outputs = []
    bindings = []
    output_idx = 0
    stream = cuda.Stream()
    for binding in engine:
        binding_dims = engine.get_binding_shape(binding)
        if len(binding_dims) == 4:
            # explicit batch case (TensorRT 7+)
            size = trt.volume(binding_dims)
        elif len(binding_dims) == 3:
            # implicit batch case (TensorRT 6 or older)
            size = trt.volume(binding_dims) * engine.max_batch_size
        else:
            raise ValueError(
                "bad dims of binding %s: %s" % (binding, str(binding_dims))
            )
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            # each grid has 3 anchors, each anchor generates a detection
            # output of 7 float32 values
            assert size % 7 == 0
            outputs.append(HostDeviceMem(host_mem, device_mem))
            output_idx += 1
    assert len(inputs) == 1
    assert len(outputs) == 1
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    """do_inference (for TensorRT 6.x or lower)

    This function is generalized for multiple inputs/outputs.
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(
        batch_size=batch_size, bindings=bindings, stream_handle=stream.handle
    )
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def do_inference_v2(context, bindings, inputs, outputs, stream):
    """do_inference_v2 (for TensorRT 7.0+)

    This function is generalized for multiple inputs/outputs for full
    dimension networks.
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


# ======================================================


@contextmanager
def cuda_context(cuda_ctx):
    try:
        cuda_ctx.push()
        yield
    finally:
        cuda_ctx.pop()


class YOLO:
    def __init__(self, engine_path: str) -> None:
        # init cuda and load yolo layer plugin
        cuda.init()
        load_yolo_plugin()
        self.cuda_ctx = cuda.Device(0).make_context()

        # tensorrt engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.engine_path = Path(engine_path)
        self._allocate_cuda()
        self.engine = self._load_engine(str(self.engine_path), self.logger)
        self.input_shape = get_input_shape(self.engine)
        # for inference
        self.do_inference = self._get_inference_function()
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(
            self.engine
        )
        self._release_cuda()

    # TensorRT Helper

    @staticmethod
    def _load_engine(engine_path: str, logger: trt.Logger):
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    @staticmethod
    def _get_inference_function():
        return do_inference if trt.__version__[0] < "7" else do_inference_v2

    # For cuda context

    def _allocate_cuda(self):
        if self.cuda_ctx:
            self.cuda_ctx.push()

    def _release_cuda(self):
        try:
            self.cuda_ctx.pop()
        except BaseException as e:
            print(f"release cuda failed: {e}")

    def _clear_inference_buffer(self):
        try:
            del self.outputs
        except BaseException:
            pass
        try:
            del self.inputs
        except BaseException:
            pass
        try:
            del self.stream
        except BaseException:
            pass

    def cleanup(self):
        """Free CUDA memories."""
        try:
            if self.outputs:
                del self.outputs
            if self.inputs:
                del self.inputs
            if self.stream:
                del self.stream
            if self.bindings:
                del self.bindings
            if self.context:
                del self.context
            if self.engine:
                del self.engine
        except BaseException as e:
            print(e)

        try:
            print("trying to clear cuda memory")
            if not self.cuda_ctx:
                print("cuda is release")
                return
            self.cuda_ctx.push()
            self.cuda_ctx.pop()
            self.cuda_ctx.detach()
        except BaseException as e:
            print(f"release cuda failed: {e}")

    # Inference

    def detect(
        self, frame: np.ndarray, threshold: float = 0.3, letter_box: bool = False
    ):
        frame_resized = _preprocess_yolo(frame, self.input_shape, letter_box)
        self.inputs[0].host = np.ascontiguousarray(frame_resized)

        with cuda_context(self.cuda_ctx):
            trt_outputs = self.do_inference(
                context=self.context,
                bindings=self.bindings,
                inputs=self.inputs,
                outputs=self.outputs,
                stream=self.stream,
            )
        boxes, scores, classes = _postprocess_yolo(
            trt_outputs,
            frame.shape[1],
            frame.shape[0],
            threshold,
            nms_threshold=0.5,
            input_shape=self.input_shape,
            letter_box=letter_box,
        )
        # clip x1, y1, x2, y2 within original image
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame.shape[1] - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame.shape[0] - 1)
        return boxes, scores, classes


class StreamYOLO:
    def __init__(
        self,
        model_path: str,
        threshold: float = 0.3,
        limit_inference_fps: int = 20,
        start_loop: bool = True,
    ):
        self.threshold = threshold

        self.yolo = YOLO(model_path)
        self.t_limit = 1 / limit_inference_fps
        self.t_prev = time.time()

        self.input = None
        self.output = ([], [], [])

        self.is_stop = False
        self.t = Thread(target=self._inference_stream, daemon=True)
        self.lock = RLock()
        if start_loop:
            self.t.start()

    def _inference_stream(self):
        print("Inference Stream is Start")
        while not self.is_stop:
            t_curr = time.time()
            if t_curr - self.t_prev < self.t_limit:
                time.sleep(0.001)
                continue
            with self.lock:
                if self.input is None:
                    continue
                self.output = self.yolo.detect(self.input, self.threshold)
            self.t_prev = time.time()

        print("Inference Streaming is Stopped")

    def stop(self):
        self.is_stop = True
        if self.t.is_alive():
            self.t.join()

    def release(self):
        self.stop()
        self.yolo.cleanup()

    def set_input(self, input):
        if input is None:
            print("get empty input")
            return
        with self.lock:
            self.input = input

    def get_output(self):
        with self.lock:
            return self.output
