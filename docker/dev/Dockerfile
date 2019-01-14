from xacc/theia-nvidia:latest
user root
run apt-get -y update \
    && unlink /usr/bin/clangd \
    && wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
    && echo "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial main" > \
                 /etc/apt/sources.list.d/llvm.list \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get -y update && apt-get install -y clang-tools-8 cmake libssl-dev \
              python3 libpython3-dev python3-pip vim gdb gfortran libblas-dev \
              liblapack-dev pkg-config libopenmpi-dev gcc-8 g++-8 gfortran-8 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 50 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 50 \
    && update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-8 50 \
    && ln -s /usr/bin/clangd-8 /usr/bin/clangd \
    && ln -s /usr/bin/clang++-8 /usr/bin/clang++ \
    && ln -s /usr/bin/clang-8 /usr/bin/clang \
    && git clone --recursive https://code.ornl.gov/qci/exatn
add settings.json /home/.theia/


