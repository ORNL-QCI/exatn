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
CMake (for build)
```

## Build instructions

Note that for now, developers must clone ExaTensor manually
``` bash
  (from top-level exatn)
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
