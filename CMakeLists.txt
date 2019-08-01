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
option(MPI_ROOT_DIR "Provide the MPI ROOT directory" "")
option (MPI_BIN_PATH "Provide the MPI bin path" "")
option(BLAS_PATH "Provide the path to the BLAS directory" "")

#Check to make sure that both the MPI implementation and root installation/bin path were supplied
if(MPI_LIB)
  if(NOT MPI_ROOT_DIR OR MPI_BIN_PATH)
    message( FATAL_ERROR "If you specify the MPI implementation, please also specify both the root path for the MPI installation. CMake is exiting.")
  endif()
else()
  if(MPI_ROOT_DIR AND MPI_BIN_PATH)
    message( FATAL_ERROR "If you don't specify the MPI implementation, please also don't specify the root and bin paths for MPI. CMake is exiting.")
  endif()
endif()

#Check to make sure that both the BLAS implementation and installation path were supplied
if(BLAS_LIB)
  if(NOT BLAS_PATH)
    message(FATAL_ERROR "If you specify the BLAS implementation, please also specify the path to the install directory. CMake is exiting.")
  endif()
else()
    if(BLAS_PATH)
      message(FATAL_ERROR "If you specify the BLAS install path, please also specify the implementation you would like to use. CMake is exiting.")
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

find_package(MPI)
find_package(OpenMP REQUIRED)
find_package(CUDAExaTN)

if(BLAS_LIB AND BLAS_PATH)
  find_package(BLAS)
endif()

message(STATUS "BLAS LIBRARIES ${BLAS_LIBRARIES}")
if(BLAS_LIB AND BLAS_PATH)
  get_filename_component(BLAS_PATH ${BLAS_LIBRARIES} DIRECTORY)
endif()
message(STATUS "BLAS path: ${BLAS_PATH}")
message(STATUS "BLAS_LIBRARIES ${BLAS_LIBRARIES}")

if(NOT CUDA_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNO_GPU")
endif()

#If BLAS implementation not user specified, proceed with "NONE"
if(NOT BLAS_LIB OR NOT BLAS_PATH)
  set(BLAS_LIB NONE)
  set(BLAS_PATH "")
endif()

set(EXA_TALSH_ONLY "NO")
#If MPI not user specified, default to no MPI
if(NOT MPI_LIB)
  set(MPI_LIB NONE)
  set(MPI_ROOT_DIR "")
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


