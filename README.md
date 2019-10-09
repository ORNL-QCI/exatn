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
Compiler (C++11, optional Fortran-2003 for multi-node execution with ExaTENSOR): GNU 8+, Intel 18+, IBM XL 16.1.1+
MPI (optional): OpenMPI 3+ (version 3.1.0 is recommended), MPICH 3+
BLAS (optional): ATLAS, MKL, ACML, ESSL
CUDA 9+ (optional)
CMake 3.9+ (for build)
```
For TaProl Parser Development
```
ANTLR: wget https://www.antlr.org/download/antlr-4.7.2-complete.jar (inside src/parser).
```

## Linux Build instructions
```
On Ubuntu 16+, for GCC 8+, OpenMPI 3+, and ATLAS BLAS, run the following:
``` bash
$ add-apt-repository ppa:ubuntu-toolchain-r/test
$ apt-get update
$ apt-get install gcc-8 g++-8 gfortran-8 libblas-dev libopenmpi-dev
$ python3 -m pip install --upgrade cmake
```

``` bash
$ git clone --recursive https://github.com/ornl-qci/exatn.git
$ cd exatn
$ mkdir build && cd build
$ cmake .. -DEXATN_BUILD_TESTS=TRUE
  For Python API:
  -DPYTHON_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['platinclude'])")
  For execution on NVIDIA GPU:
  -DENABLE_CUDA=True
  For GPU execution via very recent CUDA versions with GNU compiler:
  -DCUDA_HOST_COMPILER=<PATH_TO_CUDA_COMPATIBLE_GNU_C++_COMPILER>
  For CPU accelerated matrix algebra via a CPU BLAS library:
  -DBLAS_LIB=<BLAS_CHOICE> -DBLAS_PATH=<PATH_TO_BLAS_LIBRARIES>
   where the choices are ATLAS, MKL, ACML, ESSL.
  For multi-node execution via MPI (ExaTENSOR backend requires Fortran):
  -DMPI_LIB=<MPI_CHOICE> -DMPI_ROOT_DIR=<PATH_TO_MPI_ROOT>
   where the choices are OPENMPI or MPICH. You may also need to set
  -DMPI_BIN_PATH=<PATH_TO_MPI_BINARIES> in case they are in a different location.
$ make install

Example of a typical workstation configuration with no BLAS:
cmake ..
-DCMAKE_BUILD_TYPE=Release
-DEXATN_BUILD_TESTS=TRUE
-DPYTHON_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['platinclude'])")

Example of a typical workstation configuration with default Linux BLAS (found in /usr/lib):
cmake ..
-DCMAKE_BUILD_TYPE=Release
-DEXATN_BUILD_TESTS=TRUE
-DPYTHON_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['platinclude'])")
-DBLAS_LIB=ATLAS
-DBLAS_PATH=/usr/lib

Example of a typical workstation configuration with default Linux BLAS (found in /usr/lib) and CUDA:
cmake ..
-DCMAKE_BUILD_TYPE=Release
-DEXATN_BUILD_TESTS=TRUE
-DPYTHON_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['platinclude'])")
-DENABLE_CUDA=True -DCUDA_HOST_COMPILER=/usr/bin/g++
-DBLAS_LIB=ATLAS
-DBLAS_PATH=/usr/lib

Example of an MPI enabled configuration with default Linux BLAS (found in /usr/lib) and CUDA:
cmake ..
-DCMAKE_BUILD_TYPE=Release
-DEXATN_BUILD_TESTS=TRUE
-DPYTHON_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['platinclude'])")
-DENABLE_CUDA=True -DCUDA_HOST_COMPILER=/usr/bin/g++
-DBLAS_LIB=ATLAS
-DBLAS_PATH=/usr/lib
-DMPI_LIB=MPICH
-DMPI_ROOT_DIR=/usr/local/mpi/mpich/3.2.1
```
For GPU builds, setting the CUDA_HOST_COMPILER is necessary if your default `g++` is
not compatible with the CUDA nvcc compiler on your system. For example, CUDA 10 only
supports up to GCC 7, so if your default `g++` is version 8, then you will need to
point CMake to a compatible version (for example, g++-7 or lower, but no lower than 5).
If the build process fails to link testers at the end, make sure that
the g++ compiler used for linking tester executables is CUDA_HOST_COMPILER.

When requesting the multi-node MPI build, the tensor algebra library ExaTENSOR
is used as the default multi-node execution backend. Due to numerous bugs in
Fortran compilers and MPI libraries, the only tested choices are the following:
gcc-8 compiler, intel-18+ compiler, openmpi-3.1.0 library, mpich-3.2.1 library or later.

To use python capabilities after compilation, export the library to your `PYTHONPATH`.
```
$ export PYTHONPATH=$PYTHONPATH:~/.exatn
```
It may also be helpful to have mpi4py installed.

## Mac OS X Build Instructions (no MPI)
First install GCC via homebrew:
```
$ brew install gcc@8
```
Now continue with configuring and building ExaTN
``` bash
$ git clone --recursive https://github.com/ornl-qci/exatn.git
$ cd exatn
$ mkdir build && cd build
$ FC=gfortran-8 CXX=g++-8 cmake ..
    -DEXATN_BUILD_TESTS=TRUE
    -DPYTHON_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['platinclude'])")
$ make install
```

## Testing instructions
From build directory:
```bash
$ ctest
```

## License
See LICENSE
