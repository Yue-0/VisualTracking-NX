if [ ! -e "scripts/build.sh" ]; then
    echo "[Error] Please enter the 'Server' directory and execute:" 
    echo "        >>> bash scripts/build.sh                  "
    exit 1
fi

if [ -e "src/server/include/mot/byte_track.zip" ]; then
    cd src/server/include/mot
    echo "[INFO] Running command: unzip"
    unzip byte_track.zip
    rm byte_track.zip
    cd ../../../../
fi

if [ -d "build" ]; then
    echo "[INFO] Building packges"
    catkin_make -DCATKIN_WHITELIST_PACKAGES=""
else
    echo "[INFO] [ 1/2 ] Building srvs"
    catkin_make --only-pkg-with-deps tracking_msgs
    echo "[INFO] [ 2/2 ] Building server and client"
    catkin_make -DCATKIN_WHITELIST_PACKAGES="client;server"
fi
