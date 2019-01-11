ExaTN library: Exascale Tensor Networks

License/copyright/authors:

Description:

Dependencies:

Build instructions:
$ (from top-level exatn) mkdir build && cd build
Without ExaTENSOR:
$ cmake .. -DEXATN_BUILD_TESTS=TRUE
With ExaTENSOR:
$ cmake .. -DEXATN_BUILD_TESTS=TRUE -DEXATENSOR_ROOT=<PATH_TO_EXATENSOR>
$ make VERBOSE=1

Testing instructions:
From build directory
$ ctest (or ./src/numerics/tests/NumericsTester to run the executable)

Details:
