cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
if(${CMAKE_VERSION} VERSION_LESS 3.12)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

set(CMAKE_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_DISABLE_IN_SOURCE_BUILDS ON)
#set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(EXATENSOR_ROOT ${CMAKE_SOURCE_DIR}/tpls/ExaTensor-devel)

set(CMAKE_SKIP_INSTALL_RPATH OFF)
set(CMAKE_SKIP_RPATH OFF)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPATH_MAX=4096 -Wno-attributes")

project(exatn LANGUAGES CXX Fortran)

option(CUDA_HOST_COMPILER "Provide the host compiler for nvcc" "")
option(EXATN_BUILD_TESTS "Build ExaTN tests" OFF)
option(BLAS_LIB "Provide the BLAS implementation" "")
option(MPI_LIB "Provide the MPI implementation" "")

if(NOT CMAKE_BUILD_TYPE)
  set(
    CMAKE_BUILD_TYPE "Release"
    CACHE
      STRING
      "Choose the type of build, options are: Debug, Release, RelWithDebInfo, MinSizeRel"
    FORCE)
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.exatn"
      CACHE PATH "default install path"
      FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
include(CTest)

find_package(MPI)
find_package(OpenMP REQUIRED)
find_package(CUDAExaTN)
find_package(BLAS)


if(NOT CUDA_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNO_GPU")
endif()


message(STATUS "MPIRUN: ${MPIEXEC_EXECUTABLE}")
include_directories(${CMAKE_BINARY_DIR}/tpls/cppmicroservices/include)
include_directories(${CMAKE_BINARY_DIR}/tpls/cppmicroservices/framework/include)
include_directories(${CMAKE_SOURCE_DIR}/tpls/cppmicroservices/framework/include)

macro(exatn_enable_rpath LIBNAME)
  if(APPLE)
    set_target_properties(${LIBNAME} PROPERTIES INSTALL_RPATH "@loader_path/../lib")
    set_target_properties(${LIBNAME}
                          PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  else()
    set_target_properties(${LIBNAME} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
    set_target_properties(${LIBNAME} PROPERTIES LINK_FLAGS "-shared")
  endif()
endmacro()

configure_file("${CMAKE_SOURCE_DIR}/cmake/exatn_config.hpp.in"
               "${CMAKE_BINARY_DIR}/exatn_config.hpp")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXATN_SERVICE")

add_subdirectory(tpls)
add_subdirectory(src)

if(PYTHON_INCLUDE_DIR)
   add_subdirectory(python)
endif()

configure_file("${CMAKE_SOURCE_DIR}/cmake/exatn-config.cmake.in"
               "${CMAKE_BINARY_DIR}/exatn-config.cmake" @ONLY)
install(FILES "${CMAKE_BINARY_DIR}/exatn-config.cmake" DESTINATION .)


