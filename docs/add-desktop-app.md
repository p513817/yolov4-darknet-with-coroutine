
```bash
ROOT="~/.local/share/applications"
vim ~/.local/share/applications/cam.desktop
# chmod +wr ~/.local/share/applications/cam.desktop
```

```bash
[Desktop Entry]
Version=0.1
Exec=sudo -i /home/nvidia/workspace/jetson-orin-multicam/run.sh
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

```bash
rm $ROOT/cam.desktop
rm ~/Desktop/cam.desktop
```