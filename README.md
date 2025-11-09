for vispy
in os 
sudo apt install qtbase5-dev qt5-qmake

in venv
pip3 install PyQt5


install novnc and tigervnc
novnc from snap as github instruction

export the right display in the terminal environment, e.g., :1, the one started by tigervnc
export $DISPLAY=:1

check if the display is set correctly
echo $DISPLAY
vncserver -list

vncserver :1      

stop sceensaver
gsettings set org.gnome.desktop.screensaver lock-enabled false

use chrome to log in to novnc


On Ubuntu, numba needs tbb
pip install tbb

sudo apt-get update
sudo apt-get install libtbb-dev

(Optional) If you’re in a virtualenv
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"

Verify the Fix with numba -s

numba -s | grep TBB  

You should now see under Threading Layer Information:

TBB Threading Layer Available : True
+--> TBB imported successfully.


need the following solver in cvxpy backend
pip install highspy


show the git usage to sixi