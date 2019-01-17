# ExaTN library: Exascale Tensor Networks
ExaTN provides a software library for the expression of
hierarchical tensor networks to be leveraged in the simulation
of quantum many-body systems at exascale.

## Dependencies
```
Compiler: GCC 8+, Intel 18+, IBM XL 16.1.1+
MPI: OpenMPI, MPICH
BLAS: ATLAS, MKL, ESSL, ACML
CUDA 9+ (optional)
CMake 3.9+ (for build)
```
On Ubuntu 16.04, for GCC 8, OpenMPI, and ATLAS Blas run the following
```bash
$ add-apt-repository ppa:ubuntu-toolchain-r/test
$ apt-get update
$ apt-get install gcc-8 g++-8 gfortran-8 libblas-dev libopenmpi-dev
$ python -m pip install --upgrade cmake
```
for CMake 3.9+, do not use the apt-get installer, instead use `pip`, and
ensure that `/usr/local/bin` is in your path
```bash
$ python -m pip install --upgrade cmake
$ export PATH=$PATH:/usr/local/bin
```

## Build instructions

Note that for now, developers must clone ExaTensor manually
``` bash
$ git clone --recursive https://code.ornl.gov/qci/exatn
$ cd exatn
$ git clone https://gitlab.com/DmitryLyakh/ExaTensor tpls/ExaTensor
$ mkdir build && cd build
$ cmake .. -DEXATN_BUILD_TESTS=TRUE -DCUDA_HOST_COMPILER=$(which g++-5)
$ make
```
Setting the CUDA_HOST_COMPILER is necessary if your default `/usr/bin/g++` is
not compatible with the CUDA nvcc compiler. For example, CUDA 10 only supports up to
GCC 7, so if your default `/usr/bin/g++` is version 8, then you will need to
point CMake to a compatible version (here we use g++-5).

## Testing instructions
From build directory
```bash
$ ctest (or ./src/numerics/tests/NumericsTester to run the executable)
```

## License
See LICENSE.txt
