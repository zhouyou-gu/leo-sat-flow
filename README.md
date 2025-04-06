pip3 install PyQt5 for vispy


install novnc and tigervnc
novnc from snap as github instruction

export the right display in the terminal environment, e.g., :1, the one started by tigervnc
export $DISPLAY=:1

stop sceensaver
gsettings set org.gnome.desktop.screensaver lock-enabled false

use chrome to log in to novnc