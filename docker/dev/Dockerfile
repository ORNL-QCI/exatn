from xacc/theia-nvidia:latest
user root
run apt-get -y update \
    && unlink /usr/bin/clangd \
    && wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
    && echo "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial main" > \
                 /etc/apt/sources.list.d/llvm.list \
    && apt-get -y update && apt-get install -y clang-tools-8 cmake libssl-dev \
              python3 libpython3-dev python3-pip vim gdb gfortran libblas-dev \
              liblapack-dev pkg-config libopenmpi-dev \
    && ln -s /usr/bin/clangd-8 /usr/bin/clangd \
    && ln -s /usr/bin/clang++-8 /usr/bin/clang++ \
    && ln -s /usr/bin/clang-8 /usr/bin/clang \
    && git clone --recursive https://code.ornl.gov/qci/exatn
add settings.json /home/.theia/


