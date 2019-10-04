# Build ExaTN-enabled App with exatn-config executable

```
$ export PATH=$PATH:$HOME/.exatn/bin (or wherever exatn is installed)
$ g++ $(exatn-config --cxxflags --includes) -o test.o -c test.cpp
$ g++ test.o -o test $(exatn-config --libs)
$ ./test
```


