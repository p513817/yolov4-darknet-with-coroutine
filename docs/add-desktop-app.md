
```bash
ROOT="~/.local/share/applications"
sudo vim ~/.local/share/applications/cam.desktop
sudo chmod +wr ~/.local/share/applications/cam.desktop
```

```bash
[Desktop Entry]
Version=0.1
Exec=python3 /home/nvidia/workspace/jetson-orin-multicam/fix_main.py
Name=iCam
GenericName=iCam
Comment=Launch iCam
Terminal=true
Type=Application
Categories=Application;
```

```bash
ln -s \
~/.local/share/applications/cam.desktop \
~/Desktop/cam.desktop
```