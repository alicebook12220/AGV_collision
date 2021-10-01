#!/bin/bash
echo "AGV start"
export PATH=$PATH:~/.local/bin
export PYTHONPATH=$PYTHONPATH:/usr/local/lib
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.6/pyrealsense2
sleep 5
/usr/bin/python3 /home/auo/realsense/AGV_3d.py

function pause(){
        read -n 1 -p "$*" INP
        if [ $INP != '' ] ; then
                echo -ne '\b \n'
        fi
}

pause 'Press any key to continue...'
