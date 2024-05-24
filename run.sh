#!/bin/bash

sudo -u nvidia python3 /home/nvidia/workspace/jetson-orin-multicam/fix_main.py &
background_pid=$!
sleep 2
echo "set camera mode"
sudo i2ctransfer -f -y 9 w3@0x08 0x0A 0x42 0x08
sudo i2ctransfer -f -y 10 w3@0x08 0x0A 0x42 0x05

wait $background_pid
