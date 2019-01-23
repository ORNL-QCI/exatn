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

set(CMAKE_SKIP_INSTALL_RPATH OFF)
set(CMAKE_SKIP_RPATH OFF)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

project(exatn LANGUAGES CXX Fortran)

option(CUDA_HOST_COMPILER "Provide the host compiler for nvcc" "")
option(EXATN_BUILD_TESTS "Build ExaTN tests" OFF)

if (NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE "Release" CACHE STRING
	   "Choose the type of build, options are: Debug, Release, RelWithDebInfo, MinSizeRel"
	   FORCE
      )
endif()

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	set (CMAKE_INSTALL_PREFIX "$ENV{HOME}/.exatn" CACHE PATH "default install path" FORCE )
endif()

include(CTest)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA)

if(NOT CUDA_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNO_GPU")
endif()

if (OpenMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

include_directories(${CMAKE_BINARY_DIR}/tpls/cppmicroservices/include)
include_directories(${CMAKE_BINARY_DIR}/tpls/cppmicroservices/framework/include)

macro(exatn_enable_rpath LIBNAME)
  if(APPLE)
    set_target_properties(${LIBNAME} PROPERTIES INSTALL_RPATH "@loader_path")
    set_target_properties(${LIBNAME} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  else()
    set_target_properties(${LIBNAME} PROPERTIES INSTALL_RPATH "$ORIGIN")
    set_target_properties(${LIBNAME} PROPERTIES LINK_FLAGS "-shared")
  endif()
endmacro()

add_subdirectory(tpls)
add_subdirectory(src)

configure_file("${CMAKE_SOURCE_DIR}/cmake/exatn-config.cmake.in" "${CMAKE_BINARY_DIR}/exatn-config.cmake" @ONLY)
install(FILES "${CMAKE_BINARY_DIR}/exatn-config.cmake" DESTINATION .)

