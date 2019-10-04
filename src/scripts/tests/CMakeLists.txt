project(test CXX)
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

find_package(ExaTN REQUIRED)

add_executable(demo test.cpp)
target_link_libraries(demo PUBLIC exatn::exatn)