# Install JetPack

If your jetson without CUDA, cuDNN and TensorRT,
You can follow the command line below to install it.

```bash
# Install Jetpack directly
sudo apt install nvidia-jetpack

# Set library
vim ~/.bashrc
# in ~/.bashrc
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
source ~/.bashrc
```