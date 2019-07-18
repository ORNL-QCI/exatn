from xacc/ubuntu:18.04
run apt-get update && apt-get install -y libmpich-dev 
run git clone --recursive -b master https://github.com/ornl-qci/exatn && cd exatn && git pull && git status && mkdir build && cd build \
    && CXX=g++-8 CC=gcc-8 FC=gfortran-8 cmake .. -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DEXATN_BUILD_TESTS=TRUE \
    && make -j$(nproc) install && ctest --output-on-failure
