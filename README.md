# ExaTN library: Exascale Tensor Networks
ExaTN is a software library for the expression and processing of
hierarchical tensor networks to be leveraged in the simulation
of quantum many-body systems at exascale and other domains
which rely heavily on large-scale numerical tensor algebra.

## Dependencies
```
Compiler: GCC 8+, Intel 18+, IBM XL 16.1.1+
MPI: OpenMPI, MPICH
BLAS: ATLAS, MKL, ACML, ESSL
CUDA 9+ (optional)
CMake 3.9+ (for build)
```
On Ubuntu 16.04, for GCC 8, OpenMPI, and ATLAS Blas, run the following:
```bash
$ add-apt-repository ppa:ubuntu-toolchain-r/test
$ apt-get update
$ apt-get install gcc-8 g++-8 gfortran-8 libblas-dev libopenmpi-dev
$ python -m pip install --upgrade cmake
```
for CMake 3.9+, do not use the apt-get installer, instead use `pip`, and
ensure that `/usr/local/bin` is in your PATH:
```bash
$ python -m pip install --upgrade cmake
$ export PATH=$PATH:/usr/local/bin
```

## Build instructions

Note that, for now, developers must clone ExaTensor manually:
``` bash
$ git clone --recursive https://code.ornl.gov/qci/exatn
$ cd exatn
$ git clone https://gitlab.com/DmitryLyakh/ExaTensor tpls/ExaTensor
$ mkdir build && cd build
$ cmake .. -DEXATN_BUILD_TESTS=TRUE -DCUDA_HOST_COMPILER=<PATH_TO_CUDA_COMPATIBLE_C++_COMPILER>
  (for Python API add) -DPYTHON_INCLUDE_DIR=/usr/include/python3.5 (or wherever Python.h lives)
$ make install
```
Setting the CUDA_HOST_COMPILER is necessary if your default `g++` is not compatible
with the CUDA nvcc compiler on your system. For example, CUDA 10 only supports up to
GCC 7, so if your default `g++` is version 8, then you will need to
point CMake to a compatible version (for example, g++-5 or g++-7).

If the build process fails to link testers at the end, make sure that
the g++ compiler used for linking tester executables is CUDA_HOST_COMPILER.

## Testing instructions
From build directory:
```bash
$ ctest (or ./src/numerics/tests/NumericsTester to run the executable)
```

## License
See LICENSE.txt
