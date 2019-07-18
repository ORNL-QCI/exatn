| Branch | Status |
|:-------|:-------|
|master | [![pipeline status](https://code.ornl.gov/qci/exatn/badges/master/pipeline.svg)](https://code.ornl.gov/qci/exatn/commits/master) |
|devel | [![pipeline status](https://code.ornl.gov/qci/exatn/badges/devel/pipeline.svg)](https://code.ornl.gov/qci/exatn/commits/devel) |

# ExaTN library: Exascale Tensor Networks

ExaTN is a software library for expressing and processing
hierarchical tensor networks on homo- and heterogeneous HPC
platforms of vastly different scale, from laptops to leadership
HPC systems. The library can be leveraged in any computational
domain which heavily relies on large-scale numerical tensor algebra:
 (a) Quantum many-body theory in condensed matter physics;
 (b) Quantum many-body theory in quantum chemistry;
 (c) Quantum computing simulations;
 (d) General relativity simulations;
 (e) Multivariate data analytics;
 (f) Tensor-based neural network algorithms.


## Dependencies
```
Compiler (C++11, Fortran-2003): GNU 8+, Intel 18+, IBM XL 16.1.1+
MPI: OpenMPI 3+ (version 3.1.0 is recommended), MPICH 3+
BLAS: ATLAS, MKL, ACML, ESSL
CUDA 9+ (optional)
CMake 3.9+ (for build)
```
For TaProl Parser Development
```
ANTLR: wget https://www.antlr.org/download/antlr-4.7.2-complete.jar
```

## Linux Build instructions
```
On Ubuntu 16+, for GCC 8+, OpenMPI 3+, and ATLAS BLAS, run the following:
```bash
$ add-apt-repository ppa:ubuntu-toolchain-r/test
$ apt-get update
$ apt-get install gcc-8 g++-8 gfortran-8 libblas-dev libopenmpi-dev
$ python -m pip install --upgrade cmake
```

Note that, for now, developers must clone ExaTENSOR manually:
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
point CMake to a compatible version (for example, g++-7 or lower, but no lower than 5).
If the build process fails to link testers at the end, make sure that
the g++ compiler used for linking tester executables is CUDA_HOST_COMPILER.

## Mac OS X Build Instructions
First install MPICH or OpenMPI from source. Refer to their installation guides for this.
Here's an example configure command that we've tested for MPICH:
```
$ CC=gcc-8 CXX=g++-8 FC=gfortran-8 ./configure --prefix=/usr/local/mpich --enable-fortran=all
```
Then install GCC via homebrew (version 8 due to a bug in version 9)
```
$ brew install gcc@8
```
Now continue with configuring and building ExaTN
```
$ mkdir build && cd build
$ FC=gfortran-8 CXX=g++-8 cmake .. -DMPI_CXX_COMPILER=/usr/local/mpich/bin/mpic++ -DMPI_Fortran_COMPILER=/usr/local/mpich/bin/mpif90 -DEXATN_BUILD_TESTS=TRUE -DPYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_paths()['platinclude'])")
$ make install
```

## Testing instructions
From build directory:
```bash
$ ctest (or ./src/numerics/tests/NumericsTester to run the executable)
```

## License
See LICENSE
