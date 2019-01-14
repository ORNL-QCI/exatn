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

## Testing instructions
From build directory
```bash
$ ctest (or ./src/numerics/tests/NumericsTester to run the executable)
```

## License
See LICENSE.txt
