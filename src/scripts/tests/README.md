See test.cpp for the ExaTN-enabled application to be built.

# Build ExaTN-enabled App with exatn-config executable

```
$ export PATH=$PATH:$HOME/.exatn/bin (or wherever exatn is installed)
$ g++ $(exatn-config --cxxflags --includes) -o test.o -c test.cpp
$ g++ test.o -o test $(exatn-config --libs)
$ ./test
```

# Build ExaTN-enabled App with imported CMake Target
See CMakeLists.txt for importing ExaTN and building a target
that uses ExaTN.
```
$ mkdir build && cd build
$ cmake .. -DEXATN_DIR=$HOME/.exatn
$ make
$ ./demo
```
