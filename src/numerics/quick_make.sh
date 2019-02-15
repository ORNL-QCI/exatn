#!/bin/bash

rm *.o
g++ -c -g -fPIC -std=c++14 basis_vector.cpp
g++ -c -g -fPIC -std=c++14 space_basis.cpp
g++ -c -g -fPIC -std=c++14 spaces.cpp
g++ -c -g -fPIC -std=c++14 tensor_signature.cpp
g++ -c -g -fPIC -std=c++14 tensor_shape.cpp
g++ -c -g -fPIC -std=c++14 tensor_leg.cpp
g++ -c -g -fPIC -std=c++14 tensor.cpp
g++ -c -g -fPIC -std=c++14 tensor_exa.cpp
g++ -c -g -fPIC -std=c++14 tensor_factory.cpp
g++ -c -g -fPIC -std=c++14 register.cpp
