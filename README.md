![ExaTN](ExaTN.png)

| Branch | Status |
|:-------|:-------|
|master | [![pipeline status](https://code.ornl.gov/qci/exatn/badges/master/pipeline.svg)](https://code.ornl.gov/qci/exatn/commits/master) |
|devel | [![pipeline status](https://code.ornl.gov/qci/exatn/badges/devel/pipeline.svg)](https://code.ornl.gov/qci/exatn/commits/devel) |

# ExaTN library: Exascale Tensor Networks

ExaTN is a software library for expressing, manipulating and processing
arbitrary tensor networks on homo- and heterogeneous HPC
platforms of vastly different scale, from laptops to leadership
GPU-accelerated HPC systems. The library can be leveraged in any
computational domain that relies heavily on numerical tensor algebra:

 * Quantum many-body theory in condensed matter physics;
 * Quantum many-body theory in quantum chemistry;
 * Quantum computing simulations;
 * General relativity simulations;
 * Multivariate data analytics, tensor regression, tensor completion;
 * Tensor-based neural network algorithms.


## Concepts and Usage

[Here you can find the PDF of the ExaTN paper](https://www.frontiersin.org/articles/10.3389/fams.2022.838601/full)

The ExaTN C++ header to include is `exatn.hpp`. ExaTN provides two kinds of API:

 1. Declarative API is used to declare, construct and manipulate C++ objects
    implementing the ExaTN library concepts, like tensors, tensor networks,
    tensor network operators, tensor network expansions, etc. The corresponding
    C++ header files are located in `src/numerics`. Note that the declarative API
    calls do not allocate storage for tensors.
 2. Executive API is used to perform storage allocation and numerical processing
    of tensors, tensor networks, tensor network operators, tensor network expansions,
    etc. The corresponding header file is `src/exatn/exatn_numerics.hpp`.

There are multiple examples available in `src/exatn/tests/NumServerTester.cpp`, but you should
ignore those which use direct `numericalServer->API` calls (these are internal tests). The
`main` function at the very bottom shows how to initialize and finalize ExaTN. Note that ExaTN
assumes the column-major storage of tensors (important for initialization with external data).

Main ExaTN C++ objects:

 * `exatn::Tensor` (`src/numerics/tensor.hpp`): An abstraction of a tensor defined by
   * *Tensor name*: Alphanumeric with underscores, must begin with a letter;
   * *Tensor shape*: A vector of tensor dimension extents (extent of each tensor dimension);
   * *Tensor signature* (optional): A vector of tensor dimension identifiers. A tensor dimension
     identifier either associates the tensor dimension with a specific registered vector
     space/subspace or simply provides a base offset for defining tensor slices (default is 0).
 * `exatn::TensorNetwork` (`src/numerics/tensor_network.hpp`): A tensor network is an aggregate
   of tensors where each tensor may be connected to other tensors via associating corresponding
   tensor dimensions as specified by a directed multi-graph in which each vertex represents a
   tensor with each attached (directed) edge being a tensor dimension. Each directed edge
   connects two dimensions coming from two different tensors. Graph vertices may also have
   open edges (edges with an open end) which correspond to uncontracted tensor dimensions.
   The tensors constituting a tensor network are called *input* tensors. Each tensor network
   is also automatically equipped with the *output* tensor which collects all uncontracted
   tensor dimensions, thus representing the tensor-result of a full contraction of the
   tensor network.
 * `exatn::TensorOperator` (`src/numerics/tensor_operator.hpp`): A tensor network operator
   is a linear combination of tensor networks in which their open edges are distinguished
   by belonging to either the ket or bra tensor spaces (which do not have to be dual to each other).
 * `exatn::TensorExpansion` (`src/numerics/tensor_expansion.hpp`): A tensor network expansion
   is a linear combination of tensor networks with complex coefficients in which all open edges
   of all constituent tensor networks belong to either the ket or bra tensor space. By default,
   all open edges belong to the ket tensor space. All tensor networks in a tensor network expansion
   must have their output tensors possess the same shape (be congruent).


## API Documentation
For detailed class documentation, please see our [API Documentation](https://ornl-qci.github.io/exatn-api-docs) page.


## Dependencies
```
CMake 3.9+ (for build)
Compilers (C++14, Fortran-2003): GNU 8+
MPI (optional): MPICH 3+, OpenMPI 4+
BLAS (optional): OpenBLAS (recommended), ATLAS (default Linux BLAS), MKL, ACML (not tested), ESSL (not tested)
CUDA 9+ (optional, NVIDIA GPU only)
cuTensor/cuQuantum (optional, NVIDIA GPU only)
Note: The support of lazy tensor conjugation in ExaTN+cuQuantum is coming soon
```
For TaProl Parser Development
```
ANTLR: wget https://www.antlr.org/download/antlr-4.7.2-complete.jar (inside src/parser).
```


## Linux Build instructions
``` bash
$ git clone --recursive https://github.com/ornl-qci/exatn.git
$ cd exatn
$ git submodule init
$ git submodule update --init --recursive
$ mkdir build && cd build
$ CC=gcc CXX=g++ FC=gfortran cmake .. -DCMAKE_BUILD_TYPE=Release -DEXATN_BUILD_TESTS=TRUE

  Additional CMAKE options (BLAS, GPU, MPI, cuQuantum):

  For CPU accelerated matrix algebra via a CPU BLAS library add this:
  -DBLAS_LIB=<BLAS_CHOICE> -DBLAS_PATH=<PATH_TO_BLAS_LIBRARIES>
   where the choices are OPENBLAS, ATLAS, MKL, ACML, ESSL.
   If you use Intel MKL, you will need to provide the following
   environment variable instead of the BLAS_PATH above:
  -DPATH_INTEL_ROOT=<PATH_TO_INTEL_ROOT_DIRECTORY>

  For execution on NVIDIA GPU add this:
  -DENABLE_CUDA=True
   You can adjust the NVIDIA GPU compute capability like this:
   -DCUDA_ARCH_BIN=70
   For GPU execution via very recent CUDA versions with the GNU compiler:
   -DCUDA_HOST_COMPILER=<PATH_TO_CUDA_COMPATIBLE_GNU_C++_COMPILER>
   If you want to leverage the NVIDIA cuQuantum framework, set these:
   -DCUTENSOR=TRUE -DCUTENSOR_PATH=<PATH_TO_CUTENSOR_ROOT>
   -DCUQUANTUM=TRUE -DCUQUANTUM_PATH=<PATH_TO_CUQUANTUM_ROOT>
   Note that you will need to install cuTensor and cuQuantum
   by unpacking their Linux tar archives downloaded from NVIDIA.
   Additionally, after installing cuTensor, copy all cuTensor dynamic
   libraries (.so) from <CUTENSOR_ROOT>/lib/<CUDA_VERSION> to
   <CUTENSOR_ROOT>/lib such that the latter directory contains exactly
   the same libraries and symbolic links as the former.

  For multi-node execution via MPI add this:
  -DMPI_LIB=<MPI_CHOICE> -DMPI_ROOT_DIR=<PATH_TO_MPI_ROOT>
   where the choices are OPENMPI or MPICH. Note that the OPENMPI choice
   also covers its derivatives, for example Spectrum MPI. The MPICH choice
   also covers its derivatives, for example, Cray-MPICH. You may also need to set
  -DMPI_BIN_PATH=<PATH_TO_MPI_BINARIES> in case they are in a different location.

$ make -j install
$ make rebuild_cache
$ make install
```

Note that simply typing `make` will be insufficient and running `make install` is
mandatory, which will install all headers and libraries in the ExaTN install directory
which defaults to ~/.exatn. The install directory is the one to refer to when linking
your application with ExaTN. If you want to redefine the install directory via
CMAKE_INSTALL_PREFIX, please note that the install directory must reside outside
the ExaTN source directory. If you want to link and use ExaTN as part of your application,
the helper script located inside the ExaTN install directory `bin/exatn-config` can be used
to retrieve the necessary C++ compiler flags (`bin/exatn-config --cxxflags`),
C++ include flags (`bin/exatn-config --includes`), and C++ library linking
flags (`bin/exatn-config --libs`). The latter sometimes includes corrupted
references to some libraries, in which case you will need to examine the
generated string and manually fix it. When linking your application with
ExaTN, you should also add the generated C++ flags to the linking line.

In order to cleanly rebuild ExaTN, you will need to do the following
(from the ExaTN source root directory):
``` bash
$ rm -r ~/.exatn
$ cd ./tpls/ExaTensor
$ make clean
$ cd ../..
$ rm -r build
$ mkdir build && cd build
$ make -j install
$ make rebuild_cache
$ make install
```

```
In the following examples, replace relevant paths with your local locations.

Example of a typical workstation configuration with no BLAS (very slow):
cmake ..
-DCMAKE_BUILD_TYPE=Release
-DEXATN_BUILD_TESTS=TRUE

Example of a typical workstation configuration with default Linux BLAS (e.g. found in /usr/lib):
cmake ..
-DCMAKE_BUILD_TYPE=Release
-DEXATN_BUILD_TESTS=TRUE
-DBLAS_LIB=ATLAS -DBLAS_PATH=/usr/lib

Example of a typical workstation configuration with OpenBLAS (found in /usr/local/openblas/lib):
cmake ..
-DCMAKE_BUILD_TYPE=Release
-DEXATN_BUILD_TESTS=TRUE
-DBLAS_LIB=OPENBLAS -DBLAS_PATH=/usr/local/openblas/lib

Example of a workstation configuration with Intel MKL (with Intel root in /opt/intel):
cmake ..
-DCMAKE_BUILD_TYPE=Release
-DEXATN_BUILD_TESTS=TRUE
-DBLAS_LIB=MKL -DPATH_INTEL_ROOT=/opt/intel

Example of a typical workstation configuration with default Linux BLAS (found in /usr/lib) and CUDA:
cmake ..
-DCMAKE_BUILD_TYPE=Release
-DEXATN_BUILD_TESTS=TRUE
-DBLAS_LIB=ATLAS -DBLAS_PATH=/usr/lib
-DENABLE_CUDA=True -DCUDA_HOST_COMPILER=/usr/bin/g++ -DCUDA_ARCH_BIN=70

Example of a typical workstation configuration with OpenBLAS (found in /usr/local/openblas/lib) and CUDA:
cmake ..
-DCMAKE_BUILD_TYPE=Release
-DEXATN_BUILD_TESTS=TRUE
-DBLAS_LIB=OPENBLAS -DBLAS_PATH=/usr/local/openblas/lib
-DENABLE_CUDA=True -DCUDA_HOST_COMPILER=/usr/bin/g++ -DCUDA_ARCH_BIN=70

Example of a workstation configuration with Intel MKL (with Intel root in /opt/intel) and CUDA:
cmake ..
-DCMAKE_BUILD_TYPE=Release
-DEXATN_BUILD_TESTS=TRUE
-DBLAS_LIB=MKL -DPATH_INTEL_ROOT=/opt/intel
-DENABLE_CUDA=True -DCUDA_HOST_COMPILER=/usr/bin/g++ -DCUDA_ARCH_BIN=70

Example of an MPI-enabled configuration with default Linux BLAS (found in /usr/lib) and CUDA:
cmake ..
-DCMAKE_BUILD_TYPE=Release
-DEXATN_BUILD_TESTS=TRUE
-DBLAS_LIB=ATLAS -DBLAS_PATH=/usr/lib
-DENABLE_CUDA=True -DCUDA_HOST_COMPILER=/usr/bin/g++ -DCUDA_ARCH_BIN=70
-DMPI_LIB=MPICH -DMPI_ROOT_DIR=/usr/local/mpi/mpich/3.2.1

Example of an MPI-enabled configuration with Intel MKL (with Intel root in /opt/intel) and CUDA:
cmake ..
-DCMAKE_BUILD_TYPE=Release
-DEXATN_BUILD_TESTS=TRUE
-DBLAS_LIB=MKL -DPATH_INTEL_ROOT=/opt/intel
-DENABLE_CUDA=True -DCUDA_HOST_COMPILER=/usr/bin/g++ -DCUDA_ARCH_BIN=70
-DMPI_LIB=MPICH -DMPI_ROOT_DIR=/usr/local/mpi/mpich/3.2.1

Example of a workstation configuration with OpenBLAS, CUDA and cuQuantum:
cmake ..
-DCMAKE_BUILD_TYPE=Release
-DEXATN_BUILD_TESTS=TRUE
-DBLAS_LIB=OPENBLAS -DBLAS_PATH=/usr/local/openblas/lib
-DENABLE_CUDA=True -DCUDA_HOST_COMPILER=/usr/bin/g++ -DCUDA_ARCH_BIN=86
-DCUTENSOR=TRUE -DCUTENSOR_PATH=/usr/local/cutensor
-DCUQUANTUM=TRUE -DCUQUANTUM_PATH=/usr/local/cuquantum

Example of an MPI-enabled configuration with OpenBLAS, MPI, CUDA and cuQuantum:
cmake ..
-DCMAKE_BUILD_TYPE=Release
-DEXATN_BUILD_TESTS=TRUE
-DBLAS_LIB=OPENBLAS -DBLAS_PATH=/usr/local/openblas/lib
-DENABLE_CUDA=True -DCUDA_HOST_COMPILER=/usr/bin/g++ -DCUDA_ARCH_BIN=86
-DCUTENSOR=TRUE -DCUTENSOR_PATH=/usr/local/cutensor
-DCUQUANTUM=TRUE -DCUQUANTUM_PATH=/usr/local/cuquantum
-DMPI_LIB=OPENMPI -DMPI_ROOT_DIR=/usr/local/mpi/openmpi/4.1.2

Example of an MPI-enabled configuration with OpenBLAS and CUDA on Summit:
CC=gcc CXX=g++ FC=gfortran cmake ..
-DCMAKE_INSTALL_PREFIX=<PATH_TO_YOUR_HOME>/.exatn -DCMAKE_BUILD_TYPE=Release
-DEXATN_BUILD_TESTS=TRUE
-DBLAS_LIB=OPENBLAS -DBLAS_PATH=<PATH_TO_YOUR_OPENBLAS>/lib
-DENABLE_CUDA=True -DCUDA_HOST_COMPILER=/sw/summit/gcc/7.4.0/bin/g++ -DCUDA_ARCH_BIN=70
-DMPI_LIB=OPENMPI -DMPI_ROOT_DIR=<PATH_TO_YOUR_SPECTRUM_MPI>

On Summit, you can look up the location of libraries by "module show <MODULE_NAME>".

```
For GPU builds, setting the CUDA_HOST_COMPILER is necessary if your default `g++` is
not compatible with the CUDA nvcc compiler on your system. For example, CUDA 10 only
supports up to GCC 7, so if your default `g++` is version 8, then you will need to
point CMake to a compatible version (for example, g++-7 or lower, but no lower than 5).
If the build process fails to link testers at the end, make sure that
the g++ compiler used for linking tester executables is CUDA_HOST_COMPILER.

To use python capabilities after compilation, export the library to your `PYTHONPATH`:
```
$ export PYTHONPATH=$PYTHONPATH:~/.exatn
```
It may also be helpful to have mpi4py installed.

## Mac OS X Build Instructions (no MPI, poorly supported)
First install GCC via homebrew:
```
$ brew install gcc@8
```
Now continue with configuring and building ExaTN
``` bash
$ git clone --recursive https://github.com/ornl-qci/exatn.git
$ cd exatn
$ mkdir build && cd build
$ CC=gcc-8 CXX=g++-8 FC=gfortran-8 cmake .. -DEXATN_BUILD_TESTS=TRUE
$ make install
```

## Testing instructions
From build directory:
```bash
$ ctest
```
or, alternatively, run this (fix the number of CPU threads):
```bash
$ OMP_PLACES=cores OMP_DYNAMIC=FALSE OMP_NUM_THREADS=4 ./src/exatn/tests/NumServerTester
```
or, with MPI:
```bash
$ OMP_PLACES=cores OMP_DYNAMIC=FALSE OMP_NUM_THREADS=4 mpiexec -n 1 ./src/exatn/tests/NumServerTester
```

## License
See LICENSE (BSD-3-Clause)
