cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

set(CMAKE_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_DISABLE_IN_SOURCE_BUILDS ON)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

project(exatn LANGUAGES CXX)

option(EXATN_BUILD_TESTS "Build ExaTN tests" OFF)
option(EXATENSOR_ROOT "The root of the ExaTensor install, will contain include/lib directories" /usr)

include(CTest)

include_directories(${EXATENSOR_ROOT}/include)
link_directories(${EXATENSOR_ROOT}/lib)

find_package(CUDA)
find_package(BLAS REQUIRED)

add_subdirectory(tpls)
add_subdirectory(src)
