project(simple-test CXX)
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

find_package(ExaTN REQUIRED)

add_executable(simple simple.cpp)
target_link_libraries(simple PUBLIC exatn::exatn)
target_compile_options(simple PUBLIC "-fplugin=${EXATN_ROOT}/clang-plugins/libtaprol-syntax-handler.so")