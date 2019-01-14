# ExaTN library: Exascale Tensor Networks
ExaTN provides a software library for the expression of
hierarchical tensor networks to be leveraged in the simulation
of quantum many-body systems at exascale.

## Dependencies
GCC 8.0+, OpenMPI, CMake (for build), CUDA 9/10 (optional)

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
