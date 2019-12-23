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
set(CMAKE_DEBUG_POSTFIX "")

set(CMAKE_SKIP_INSTALL_RPATH OFF)
set(CMAKE_SKIP_RPATH OFF)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPATH_MAX=4096 -Wno-attributes")

project(exatn LANGUAGES CXX Fortran)

option(EXATN_BUILD_TESTS "Build ExaTN tests" OFF)
option(CUDA_HOST_COMPILER "Provide the host compiler for nvcc" "")
option(BLAS_LIB "Provide the BLAS implementation (ATLAS,MKL,ACML,ESSL)" "")
option(BLAS_PATH "Provide the path to the BLAS libraries" "")
option(WITH_LAPACK "Turn on LAPACK support" "FALSE")
option(MPI_LIB "Provide the MPI implementation (MPICH,OPENMPI)" "")
option(MPI_ROOT_DIR "Provide the MPI root directory" "")
option(MPI_BIN_PATH "Provide the MPI bin path" "")
option(ENABLE_CUDA "Turn on CUDA support" OFF)

#Check to make sure that both the MPI implementation and root installation/bin path were supplied
if(MPI_LIB)

  if(NOT MPI_LIB STREQUAL "MPICH" AND NOT MPI_LIB STREQUAL "OPENMPI")
    message(FATAL_ERROR "Invalid MPI_LIB choice: Available choices: MPICH, OPENMPI. CMake is exiting.")
  endif()

  if(NOT MPI_ROOT_DIR)
    message(FATAL_ERROR "If you specify the MPI implementation, please also specify the MPI root path (-DMPI_ROOT_DIR). CMake is exiting.")
  endif()

  if(NOT MPI_BIN_PATH)
    set(MPI_BIN_PATH ${MPI_ROOT_DIR}/bin)
  endif()

else()

  if(MPI_ROOT_DIR OR MPI_BIN_PATH)
    message(FATAL_ERROR "If you don't specify the MPI implementation, please also don't specify the MPI root and/or bin paths. CMake is exiting.")
  endif()

endif()

#Check to make sure that both the BLAS implementation and installation path were supplied
if(BLAS_LIB)

  if(NOT BLAS_PATH AND NOT BLAS_LIB STREQUAL "MKL")
    message(FATAL_ERROR "If you specify the BLAS implementation, you must also specify the path to the BLAS libraries (-DBLAS_PATH). CMake is exiting.")
  endif()

  if(BLAS_LIB STREQUAL "ATLAS")
    if(NOT EXISTS "${BLAS_PATH}/libblas.so" AND NOT EXISTS "${BLAS_PATH}/libblas.a")
      message(FATAL_ERROR "The ATLAS BLAS library path does not contain libblas. CMake is exiting.")
    endif()
  elseif(BLAS_LIB STREQUAL "MKL")
    if(NOT PATH_INTEL_ROOT)
      message(FATAL_ERROR "The MKL BLAS library choice requires the Intel root directory (-DPATH_INTEL_ROOT), for example /opt/intel. CMake is exiting.")
    endif()
    set(BLAS_PATH ${PATH_INTEL_ROOT}/mkl/lib/intel64)
    if(NOT EXISTS "${BLAS_PATH}/libmkl_core.so")
      message(FATAL_ERROR "The MKL BLAS library path does not contain libmkl_core.so. CMake is exiting.")
    endif()
  elseif(BLAS_LIB STREQUAL "ACML")
    if(NOT EXISTS "${BLAS_PATH}/libacml_mp.so")
      message(FATAL_ERROR "The ACML BLAS library path does not contain libacml_mp.so. CMake is exiting.")
    endif()
  elseif(BLAS_LIB STREQUAL "ESSL")
    if(NOT EXISTS "${BLAS_PATH}/libessl.so")
      message(FATAL_ERROR "The ESSL BLAS library path does not contain libessl.so. CMake is exiting.")
    endif()
    if(NOT PATH_IBM_XL_CPP)
      message(FATAL_ERROR "The ESSL BLAS choice requires the path to IBM XL C++ runtime libraries (-DPATH_IBM_XL_CPP). CMake is exiting.")
    endif()
    if(NOT PATH_IBM_XL_FOR)
      message(FATAL_ERROR "The ESSL BLAS choice requires the path to IBM XL Fortran runtime libraries (-DPATH_IBM_XL_FOR). CMake is exiting.")
    endif()
    if(NOT PATH_IBM_XL_SMP)
      message(FATAL_ERROR "The ESSL BLAS choice requires the path to IBM XL SMP runtime libraries (-DPATH_IBM_XL_SMP). CMake is exiting.")
    endif()
  else()
    message(FATAL_ERROR "Invalid BLAS_LIB choice: Available choices: ATLAS, MKL, ACML, ESSL. CMake is exiting.")
  endif()

else()

  if(BLAS_PATH)
    message(FATAL_ERROR "If you specify the path to the BLAS libraries, please also specify which BLAS it is: ATLAS, MKL, ACML, ESSL. CMake is exiting.")
  endif()

endif()

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

find_package(OpenMP REQUIRED)

find_package(MPI)

if(ENABLE_CUDA)
  find_package(CUDAExaTN)
  if(NOT CUDA_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNO_GPU")
    message(WARNING "You specified ENABLE_CUDA=TRUE but find_package(CUDA) could not find CUDA development headers or libraries")
  endif()
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNO_GPU")
  set(CUDA_FOUND FALSE)
endif()

#If BLAS implementation not user specified, proceed with "NONE"
if(NOT BLAS_LIB)
  set(BLAS_LIB NONE)
  set(BLAS_PATH "")
endif()

#If MPI implementation not user specified, proceed with "NONE"
set(EXA_TALSH_ONLY "NO")
if(NOT MPI_LIB)
  set(MPI_LIB NONE)
  set(MPI_ROOT_DIR "")
  set(MPI_BIN_PATH "")
  set(EXA_TALSH_ONLY "YES")
endif()

message(STATUS "MPIRUN: ${MPIEXEC_EXECUTABLE}")
include_directories(${CMAKE_BINARY_DIR}/tpls/cppmicroservices/include)
include_directories(${CMAKE_BINARY_DIR}/tpls/cppmicroservices/framework/include)
include_directories(${CMAKE_SOURCE_DIR}/tpls/cppmicroservices/framework/include)

macro(exatn_configure_library_rpath LIBNAME)
  if(APPLE)
    set_target_properties(${LIBNAME} PROPERTIES INSTALL_RPATH "@loader_path")
    set_target_properties(${LIBNAME}
                          PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  else()
    set_target_properties(${LIBNAME} PROPERTIES INSTALL_RPATH "$ORIGIN")
    set_target_properties(${LIBNAME} PROPERTIES LINK_FLAGS "-shared")
  endif()
endmacro()
macro(exatn_configure_plugin_rpath LIBNAME)
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
