# ExaTN library: Exascale Tensor Networks
ExaTN provides a software library for the expression of
hierarchical tensor networks to be leveraged in the simulation
of quantum many-body systems at exascale.

## Dependencies
GCC 8.0+, CUDA 9/10, MPI, CMake (for build)

## Build instructions

``` bash
$ (from top-level exatn) mkdir build && cd build
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
