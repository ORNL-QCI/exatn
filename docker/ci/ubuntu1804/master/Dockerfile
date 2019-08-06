from xacc/ubuntu:18.04
run git clone --recursive -b master https://github.com/ornl-qci/exatn \
    && cd exatn && git pull && git status && mkdir build && cd build \
    && CXX=g++-8 CC=gcc-8 FC=gfortran-8 \
       cmake .. -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
                -DEXATN_BUILD_TESTS=TRUE -DBLAS_LIB=ATLAS \
                -DBLAS_PATH=/usr/lib/x86_64-linux-gnu \
    && make -j$(nproc) install && ctest --output-on-failure
