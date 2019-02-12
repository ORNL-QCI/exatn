from xacc/theia-nvidia:latest
user root
run apt-get -y update \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get -y update && apt-get install -y libssl-dev \
              python3 libpython3-dev python3-pip vim gdb gfortran libblas-dev \
              liblapack-dev pkg-config libopenmpi-dev gcc-8 g++-8 gfortran-8 \
    && python3 -m pip install cmake \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 50 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 50 \
    && update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-8 50 
add settings.json /home/.theia/


