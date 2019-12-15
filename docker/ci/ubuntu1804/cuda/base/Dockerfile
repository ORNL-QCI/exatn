from nvidia/cuda:10.0-devel
run apt-get -y update && apt-get install -y gcc-8 g++-8 gfortran-8 git wget \
            python3 libunwind-dev libmpich-dev \
            libpython3-dev python3-pip libblas-dev liblapack-dev software-properties-common \
    && python3 -m pip install cmake \
    && rm -rf /var/lib/apt/lists/* 
