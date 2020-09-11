To compile and run simple.cpp

```bash
$ mkdir build && cd build 
$ cmake .. -DCMAKE_CXX_COMPILER=/path/to/syntax/handler/clang++ -DExaTN_DIR=$HOME/.exatn
$ make 
$ ./simple
```