set(BUILD_SHARED_LIBS TRUE)
include_directories(${CMAKE_BINARY_DIR}/tpls/cppmicroservices/include)
include_directories(${CMAKE_BINARY_DIR}/tpls/cppmicroservices/framework/include)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "9.0.0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-copy")
  endif()
endif()

add_subdirectory(cppmicroservices)

find_package(Git QUIET)

if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
  execute_process(COMMAND ${GIT_EXECUTABLE}
                          submodule
                          update
                          --init
                          --recursive
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  RESULT_VARIABLE GIT_SUBMOD_RESULT)
  if(NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(
      FATAL_ERROR
        "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules"
      )
  endif()
endif()

if(EXATN_BUILD_TESTS)
  enable_testing()

  add_subdirectory(gtest)
  mark_as_advanced(BUILD_GMOCK
                   BUILD_GTEST
                   BUILD_SHARED_LIBS
                   gmock_build_tests
                   gtest_build_samples
                   gtest_build_tests
                   gtest_disable_pthreads
                   gtest_force_shared_crt
                   gtest_hide_internal_symbols)

  macro(exatn_add_test TESTNAME)
    add_executable(${TESTNAME} ${ARGN})
    target_link_libraries(${TESTNAME} PRIVATE gtest gmock gtest_main)
    add_test(${TESTNAME} ${TESTNAME})
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
  endmacro()
endif()

set(TALSHXX_EXISTS "NO")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ExaTensor/include/talshxx.hpp")
  set(TALSHXX_EXISTS "YES")
endif()

set(EXA_TALSH_ONLY "NO")
if(NOT MPI_LIB OR MPI_LIB STREQUAL "NONE")
  set(EXA_TALSH_ONLY "YES")
endif()

if(NOT CUDA_HOST_COMPILER)
  set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
endif()

message(STATUS "The MPI implementation is set to ${MPI_LIB}")
message(STATUS "The MPI root path is set to ${MPI_ROOT_DIR}")
message(STATUS "The MPI bin path is set to ${MPI_BIN_PATH}")
message(STATUS "The BLAS implementation is set to ${BLAS_LIB}")
message(STATUS "The BLAS library path is set to ${BLAS_PATH}")
message(STATUS "The CUDA Host compiler is set to ${CUDA_HOST_COMPILER}")
message(STATUS "Solely TAL-SH backend single-node build: ${EXA_TALSH_ONLY}")

if(APPLE)
  message(STATUS "This is Mac OS build, building TAL-SH CPU backend only")

  if(BLAS_LIB STREQUAL "ACML" OR BLAS_LIB STREQUAL "ESSL")
    message(FATAL_ERROR "Mac OS build does not support ACML or ESSL BLAS libraries!")
  endif()

  if(NOT MPI_LIB OR MPI_LIB STREQUAL "NONE")
    add_custom_target(exatensor-build
                      COMMAND ${CMAKE_COMMAND}
                            -E
                            env
                            CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                            CMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                            CMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
                            GPU_CUDA=NOCUDA
                            MPILIB=${MPI_LIB}
                            BLASLIB=${BLAS_LIB}
                            EXA_NO_BUILD=${TALSHXX_EXISTS}
                            EXA_TALSH_ONLY=YES
                            EXATN_SERVICE=YES
                            EXA_OS=NO_LINUX
                            PATH_BLAS_${BLAS_LIB}=${BLAS_PATH}
                            PATH_INTEL=${PATH_INTEL_ROOT}
                            make
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ExaTensor)
  elseif(MPI_LIB STREQUAL "OPENMPI")
    add_custom_target(exatensor-build
                      COMMAND ${CMAKE_COMMAND}
                            -E
                            env
                            OMPI_CXX=${CMAKE_CXX_COMPILER}
                            OMPI_CC=${CMAKE_C_COMPILER}
                            OMPI_FC=${CMAKE_Fortran_COMPILER}
                            GPU_CUDA=NOCUDA
                            MPILIB=${MPI_LIB}
                            BLASLIB=${BLAS_LIB}
                            EXA_NO_BUILD=${TALSHXX_EXISTS}
                            PATH_${MPI_LIB}=${MPI_ROOT_DIR}
                            EXA_TALSH_ONLY=NO
                            EXATN_SERVICE=YES
                            EXA_OS=NO_LINUX
                            PATH_${MPI_LIB}_BIN=${MPI_BIN_PATH}
                            PATH_BLAS_${BLAS_LIB}=${BLAS_PATH}
                            PATH_INTEL=${PATH_INTEL_ROOT}
                            make
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ExaTensor)
  elseif(MPI_LIB STREQUAL "MPICH")
    add_custom_target(exatensor-build
                      COMMAND ${CMAKE_COMMAND}
                            -E
                            env
                            MPICH_CXX=${CMAKE_CXX_COMPILER}
                            MPICH_CC=${CMAKE_C_COMPILER}
                            MPICH_FC=${CMAKE_Fortran_COMPILER}
                            GPU_CUDA=NOCUDA
                            MPILIB=${MPI_LIB}
                            BLASLIB=${BLAS_LIB}
                            EXA_NO_BUILD=${TALSHXX_EXISTS}
                            PATH_${MPI_LIB}=${MPI_ROOT_DIR}
                            EXA_TALSH_ONLY=NO
                            EXATN_SERVICE=YES
                            EXA_OS=NO_LINUX
                            PATH_${MPI_LIB}_BIN=${MPI_BIN_PATH}
                            PATH_BLAS_${BLAS_LIB}=${BLAS_PATH}
                            PATH_INTEL=${PATH_INTEL_ROOT}
                            make
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ExaTensor)
  endif()

else()

  if(CUDA_FOUND)
    message(STATUS "CUDA ROOT: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA LIBRARIES: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA BLAS LIBRARIES: ${CUDA_CUBLAS_LIBRARIES}")

    if(NOT MPI_LIB OR MPI_LIB STREQUAL "NONE")
      add_custom_target(exatensor-build
                        COMMAND ${CMAKE_COMMAND}
                              -E
                              env
                              CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                              CMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                              CMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
                              GPU_CUDA=CUDA
                              PATH_CUDA=${CUDA_TOOLKIT_ROOT_DIR}
                              CUDA_HOST_COMPILER=${CUDA_HOST_COMPILER}
                              MPILIB=${MPI_LIB}
                              BLASLIB=${BLAS_LIB}
                              EXA_NO_BUILD=${TALSHXX_EXISTS}
                              EXA_TALSH_ONLY=YES
                              EXATN_SERVICE=YES
                              PATH_BLAS_${BLAS_LIB}=${BLAS_PATH}
                              PATH_INTEL=${PATH_INTEL_ROOT}
                              make
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ExaTensor)
    elseif(MPI_LIB STREQUAL "OPENMPI")
      add_custom_target(exatensor-build
                        COMMAND ${CMAKE_COMMAND}
                              -E
                              env
                              OMPI_CXX=${CMAKE_CXX_COMPILER}
                              OMPI_CC=${CMAKE_C_COMPILER}
                              OMPI_FC=${CMAKE_Fortran_COMPILER}
                              GPU_CUDA=CUDA
                              PATH_CUDA=${CUDA_TOOLKIT_ROOT_DIR}
                              CUDA_HOST_COMPILER=${CUDA_HOST_COMPILER}
                              MPILIB=${MPI_LIB}
                              BLASLIB=${BLAS_LIB}
                              EXA_NO_BUILD=${TALSHXX_EXISTS}
                              PATH_${MPI_LIB}=${MPI_ROOT_DIR}
                              EXA_TALSH_ONLY=NO
                              EXATN_SERVICE=YES
                              PATH_${MPI_LIB}_BIN=${MPI_BIN_PATH}
                              PATH_BLAS_${BLAS_LIB}=${BLAS_PATH}
                              PATH_INTEL=${PATH_INTEL_ROOT}
                              make
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ExaTensor)
    elseif(MPI_LIB STREQUAL "MPICH")
      add_custom_target(exatensor-build
                        COMMAND ${CMAKE_COMMAND}
                              -E
                              env
                              MPICH_CXX=${CMAKE_CXX_COMPILER}
                              MPICH_CC=${CMAKE_C_COMPILER}
                              MPICH_FC=${CMAKE_Fortran_COMPILER}
                              GPU_CUDA=CUDA
                              PATH_CUDA=${CUDA_TOOLKIT_ROOT_DIR}
                              CUDA_HOST_COMPILER=${CUDA_HOST_COMPILER}
                              MPILIB=${MPI_LIB}
                              BLASLIB=${BLAS_LIB}
                              EXA_NO_BUILD=${TALSHXX_EXISTS}
                              PATH_${MPI_LIB}=${MPI_ROOT_DIR}
                              EXA_TALSH_ONLY=NO
                              EXATN_SERVICE=YES
                              PATH_${MPI_LIB}_BIN=${MPI_BIN_PATH}
                              PATH_BLAS_${BLAS_LIB}=${BLAS_PATH}
                              PATH_INTEL=${PATH_INTEL_ROOT}
                              make
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ExaTensor)
    endif()

  else()

    if(NOT MPI_LIB OR MPI_LIB STREQUAL "NONE")
      add_custom_target(exatensor-build
                        COMMAND ${CMAKE_COMMAND}
                              -E
                              env
                              CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                              CMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                              CMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
                              GPU_CUDA=NOCUDA
                              MPILIB=${MPI_LIB}
                              BLASLIB=${BLAS_LIB}
                              EXA_NO_BUILD=${TALSHXX_EXISTS}
                              EXA_TALSH_ONLY=YES
                              EXATN_SERVICE=YES
                              PATH_BLAS_${BLAS_LIB}=${BLAS_PATH}
                              PATH_INTEL=${PATH_INTEL_ROOT}
                              make
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ExaTensor)
    elseif(MPI_LIB STREQUAL "OPENMPI")
      add_custom_target(exatensor-build
                        COMMAND ${CMAKE_COMMAND}
                              -E
                              env
                              OMPI_CXX=${CMAKE_CXX_COMPILER}
                              OMPI_CC=${CMAKE_C_COMPILER}
                              OMPI_FC=${CMAKE_Fortran_COMPILER}
                              GPU_CUDA=NOCUDA
                              MPILIB=${MPI_LIB}
                              BLASLIB=${BLAS_LIB}
                              EXA_NO_BUILD=${TALSHXX_EXISTS}
                              PATH_${MPI_LIB}=${MPI_ROOT_DIR}
                              EXA_TALSH_ONLY=NO
                              EXATN_SERVICE=YES
                              PATH_${MPI_LIB}_BIN=${MPI_BIN_PATH}
                              PATH_BLAS_${BLAS_LIB}=${BLAS_PATH}
                              PATH_INTEL=${PATH_INTEL_ROOT}
                              make
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ExaTensor)
    elseif(MPI_LIB STREQUAL "MPICH")
      add_custom_target(exatensor-build
                        COMMAND ${CMAKE_COMMAND}
                              -E
                              env
                              MPICH_CXX=${CMAKE_CXX_COMPILER}
                              MPICH_CC=${CMAKE_C_COMPILER}
                              MPICH_FC=${CMAKE_Fortran_COMPILER}
                              GPU_CUDA=NOCUDA
                              MPILIB=${MPI_LIB}
                              BLASLIB=${BLAS_LIB}
                              EXA_NO_BUILD=${TALSHXX_EXISTS}
                              PATH_${MPI_LIB}=${MPI_ROOT_DIR}
                              EXA_TALSH_ONLY=NO
                              EXATN_SERVICE=YES
                              PATH_${MPI_LIB}_BIN=${MPI_BIN_PATH}
                              PATH_BLAS_${BLAS_LIB}=${BLAS_PATH}
                              PATH_INTEL=${PATH_INTEL_ROOT}
                              make
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ExaTensor)
    endif()

  endif()
endif()

# Compiler-specific C++11 activation (FROM ANTLR4).
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU"
   OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
                  OUTPUT_VARIABLE GCC_VERSION)
  # Just g++-5.0 and greater contain <codecvt> header. (test in ubuntu)
  if(NOT (GCC_VERSION VERSION_GREATER 5.0 OR GCC_VERSION VERSION_EQUAL 5.0))
    message(
      WARNING
        "${PROJECT_NAME} requires g++ 5.0 or greater. XACC will not build Antlr support."
      )
  else()
    add_subdirectory(antlr)
  endif()
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
  add_subdirectory(antlr)
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang"
       AND CMAKE_SYSTEM_NAME MATCHES "Linux")
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
                  OUTPUT_VARIABLE CLANG_VERSION)
  if(NOT
     (CLANG_VERSION VERSION_GREATER 4.2.1 OR CLANG_VERSION VERSION_EQUAL 4.2.1))
    message(
      WARNING
        "${PROJECT_NAME} requires clang 4.2.1 or greater. XACC will not build Antlr support."
      )
  else()
    add_subdirectory(antlr)
  endif()
  # You can use libc++ to compile this project when g++ is NOT greater than or equal to 5.0.
  if(WITH_LIBCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
  endif()
endif()

set(BUILD_SHARED_LIBS FALSE)
set(BOOST_LIBS_OPTIONAL graph CACHE STRING "" FORCE)
add_subdirectory(boost-cmake)

add_custom_command(TARGET exatensor-build POST_BUILD
                   COMMAND ${CMAKE_COMMAND}
                           -E
                           make_directory
                           ${CMAKE_INSTALL_PREFIX}/lib
                   COMMAND ${CMAKE_COMMAND}
                           -E
                           copy
                           ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/lib/lib*
                           ${CMAKE_INSTALL_PREFIX}/lib/
                   WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                   COMMENT "Installing ExaTensor Libs to Install Prefix...")
add_custom_command(
  TARGET exatensor-build POST_BUILD
  COMMAND ${CMAKE_COMMAND}
          -E
          copy
          ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include/tensor_method.hpp
          ${CMAKE_INSTALL_PREFIX}/include/exatensor
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Installing ExaTensor tensor_method.hpp to Install Prefix...")
