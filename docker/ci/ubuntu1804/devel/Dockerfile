from code.ornl.gov:4567/qci/exatn/base_cuda:10.0
run update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 50 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 50 \
    && update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-8 50 \
    && gcc --version && g++ --version && gfortran --version \
    && git clone --recursive https://github.com/ornl-qci/exatn \
    && cd exatn && git pull && git status && mkdir build && cd build \
    && cmake .. -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
                -DEXATN_BUILD_TESTS=TRUE -DBLAS_LIB=ATLAS \
                -DBLAS_PATH=/usr/lib/x86_64-linux-gnu \
                -DMPI_LIB=MPICH -DMPI_ROOT_DIR=/usr/lib/mpich \
                -DMPI_BIN_PATH=/usr/bin -DCUDA_HOST_COMPILER=g++-7 \
    && make -j$(nproc) install 
