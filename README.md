# YOLOv4 TensorRT Sample

## Features
* Support tensorrt engine from darknet yolo.
* Support inference and capturing camera with asynchronous mode.

## Tested Device
<details>
<summary><code>Jetson Orin Nano 8G</code></summary>
<br>

```bash 
Software part of jetson-stats 4.2.8 - (c) 2024, Raffaello Bonghi
Model: NVIDIA Orin Nano Developer Kit - Jetpack 5.1.2 [L4T 35.4.1]
NV Power Mode[0]: 15W
Hardware:
- P-Number: p3767-0003
- Module: NVIDIA Jetson Orin Nano (8GB ram)
Platform:
- Distribution: Ubuntu 20.04 focal
- Release: 5.10.120-tegra
jtop:
- Version: 4.2.8
- Service: Active
Libraries:
- CUDA: 11.4.315
- cuDNN: 8.6.0.166
- TensorRT: 8.5.2.2
- VPI: 2.3.9
- Vulkan: 1.3.204
- OpenCV: 4.5.4 - with CUDA: NO
```
</details>

## Environment
* NOTE: If your jetson without CUDA, cuDNN and TensorRT, please follow this [document](./docs/install-jetpack.md) to install it.
```bash
cd yolo

# Pycuda
./install_pycuda.sh

# ONNX
sudo apt-get install protobuf-compiler libprotoc-dev
sudo pip3 install onnx==1.9.0
```

## Download & Convert YOLOv4
```bash
cd yolo
./download_yolo.sh
python3 yolo_to_onnx.py -m yolov4-tiny-416
python3 onnx_to_tensorrt.py -m yolov4-tiny-416 # about 4 min
```

## Run
```bash
python3 main.py
```

## Reference
* [tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos/tree/master)
* [jetson install onnx](https://forums.developer.nvidia.com/t/problem-with-installing-onnx-on-jetson-nano/110820/12)
* 


git config --global user.email "p513817@gmail.com" \
git config --global user.name "p513817"