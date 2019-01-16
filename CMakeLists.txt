cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

set(CMAKE_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_DISABLE_IN_SOURCE_BUILDS ON)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(EXATENSOR_ROOT ${CMAKE_SOURCE_DIR}/tpls/ExaTensor-devel)

project(exatn LANGUAGES CXX Fortran)

option(CUDA_HOST_COMPILER "Provide the host compiler for nvcc" "")
option(EXATN_BUILD_TESTS "Build ExaTN tests" OFF)

include(CTest)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA)

message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")

if (OpenMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fopenmp")
endif()

link_directories(${CMAKE_SOURCE_DIR}/tpls/ExaTensor/lib)

add_subdirectory(tpls)
add_subdirectory(src)


