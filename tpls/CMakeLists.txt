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

  if(MPI_LIB AND NOT MPI_LIB STREQUAL "NONE")

    macro(exatn_add_mpi_test TESTNAME)
      add_executable(${TESTNAME} ${ARGN})
      target_link_libraries(${TESTNAME} PRIVATE gtest gmock gtest_main)
      set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
      get_filename_component(MPI_BIN_PATH ${MPI_CXX_COMPILER} DIRECTORY)
      add_test(NAME ${TESTNAME} COMMAND ${MPI_BIN_PATH}/mpiexec -np 2 ./${TESTNAME})
    endmacro()

  else()

    macro(exatn_add_mpi_test TESTNAME)
      add_executable(${TESTNAME} ${ARGN})
      target_link_libraries(${TESTNAME} PRIVATE gtest gmock gtest_main)
      add_test(${TESTNAME} ${TESTNAME})
      set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
    endmacro()

  endif()

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

if(WITH_LAPACK)
  set(LAPACK_LIBRARIES "")
  if    (BLAS_LIB STREQUAL "ATLAS")
    set(LAPACK_LIBRARIES "-llapack")
  elseif(BLAS_LIB STREQUAL "MKL")
  elseif(BLAS_LIB STREQUAL "OPENBLAS")
  elseif(BLAS_LIB STREQUAL "ACML")
  elseif(BLAS_LIB STREQUAL "ESSL")
    set(LAPACK_LIBRARIES "${essl};${xlf90_r};${xlfmath}")
  endif()
endif()


message(STATUS "The MPI implementation is set to ${MPI_LIB}")
message(STATUS "The MPI root path is set to ${MPI_ROOT_DIR}")
message(STATUS "The MPI bin path is set to ${MPI_BIN_PATH}")
message(STATUS "The BLAS implementation is set to ${BLAS_LIB}")
message(STATUS "The BLAS library path is set to ${BLAS_PATH}")
message(STATUS "The BLAS library must also have LAPACK: ${WITH_LAPACK}")
message(STATUS "The CUDA Host compiler is set to ${CUDA_HOST_COMPILER}")
message(STATUS "Solely TAL-SH backend single-node build: ${EXA_TALSH_ONLY}")


if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  set(TOOLKIT "GNU")
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
  set(TOOLKIT "INTEL")
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Cray")
  set(TOOLKIT "CRAY")
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "IBM")
  set(TOOLKIT "IBM")
else()
  message(FATAL_ERROR "Your discovered compiler suite ${CMAKE_CXX_COMPILER_ID} is not supported. CMake is exiting.")
endif()


set(EXATENSOR_BUILD_TYPE OPT)
if (CMAKE_BUILD_TYPE MATCHES "Debug" OR CMAKE_BUILD_TYPE MATCHES "DEBUG")
 message(STATUS "Setting ExaTensor BUILD_TYPE to DEV")
 set(EXATENSOR_BUILD_TYPE DEV)
endif()

if(APPLE)
  message(STATUS "This is Mac OS build, building the TAL-SH CPU backend only")

  if(BLAS_LIB STREQUAL "ACML" OR BLAS_LIB STREQUAL "ESSL")
    message(FATAL_ERROR "Mac OS build does not support ACML or ESSL BLAS libraries. CMake is exiting.")
  endif()

  if(NOT MPI_LIB OR MPI_LIB STREQUAL "NONE")
    add_custom_target(exatensor-build
                      COMMAND ${CMAKE_COMMAND}
                            -E
                            env
                            CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                            CMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                            CMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
                            BUILD_TYPE=${EXATENSOR_BUILD_TYPE}
                            TOOLKIT=${TOOLKIT}
                            EXA_NO_BUILD=${TALSHXX_EXISTS}
                            EXA_TALSH_ONLY=YES
                            EXATN_SERVICE=YES
                            EXA_OS=NO_LINUX
                            GPU_CUDA=NOCUDA
                            MPILIB=NONE
                            BLASLIB=${BLAS_LIB}
                            PATH_BLAS_${BLAS_LIB}=${BLAS_PATH}
                            PATH_INTEL=${PATH_INTEL_ROOT}
                            WITH_LAPACK=${WITH_LAPACK}
                            PATH_LAPACK_LIB=${BLAS_PATH}
                            LAPACK_LIBS="${LAPACK_LIBRARIES}"
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
                            TOOLKIT=${TOOLKIT}
                            BUILD_TYPE=${EXATENSOR_BUILD_TYPE}
                            EXA_NO_BUILD=${TALSHXX_EXISTS}
                            EXA_TALSH_ONLY=NO
                            EXATN_SERVICE=YES
                            EXA_OS=NO_LINUX
                            GPU_CUDA=NOCUDA
                            MPILIB=${MPI_LIB}
                            PATH_${MPI_LIB}=${MPI_ROOT_DIR}
                            PATH_${MPI_LIB}_BIN=${MPI_BIN_PATH}
                            BLASLIB=${BLAS_LIB}
                            PATH_BLAS_${BLAS_LIB}=${BLAS_PATH}
                            PATH_INTEL=${PATH_INTEL_ROOT}
                            WITH_LAPACK=${WITH_LAPACK}
                            PATH_LAPACK_LIB=${BLAS_PATH}
                            LAPACK_LIBS="${LAPACK_LIBRARIES}"
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
                            TOOLKIT=${TOOLKIT}
                            BUILD_TYPE=${EXATENSOR_BUILD_TYPE}
                            EXA_NO_BUILD=${TALSHXX_EXISTS}
                            EXA_TALSH_ONLY=NO
                            EXATN_SERVICE=YES
                            EXA_OS=NO_LINUX
                            GPU_CUDA=NOCUDA
                            MPILIB=${MPI_LIB}
                            PATH_${MPI_LIB}=${MPI_ROOT_DIR}
                            PATH_${MPI_LIB}_BIN=${MPI_BIN_PATH}
                            BLASLIB=${BLAS_LIB}
                            PATH_BLAS_${BLAS_LIB}=${BLAS_PATH}
                            PATH_INTEL=${PATH_INTEL_ROOT}
                            WITH_LAPACK=${WITH_LAPACK}
                            PATH_LAPACK_LIB=${BLAS_PATH}
                            LAPACK_LIBS="${LAPACK_LIBRARIES}"
                            make
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ExaTensor)
  endif()

else()

  if(CUDA_FOUND)
    message(STATUS "This is Linux build with GPU CUDA support")
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
                              TOOLKIT=${TOOLKIT}
                              BUILD_TYPE=${EXATENSOR_BUILD_TYPE}
                              EXA_NO_BUILD=${TALSHXX_EXISTS}
                              EXA_TALSH_ONLY=YES
                              EXATN_SERVICE=YES
                              EXA_OS=LINUX
                              GPU_CUDA=CUDA
                              PATH_CUDA=${CUDA_TOOLKIT_ROOT_DIR}
                              CUDA_HOST_COMPILER=${CUDA_HOST_COMPILER}
                              MPILIB=NONE
                              BLASLIB=${BLAS_LIB}
                              PATH_BLAS_${BLAS_LIB}=${BLAS_PATH}
                              PATH_INTEL=${PATH_INTEL_ROOT}
                              WITH_LAPACK=${WITH_LAPACK}
                              PATH_LAPACK_LIB=${BLAS_PATH}
                              LAPACK_LIBS="${LAPACK_LIBRARIES}"
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
                              TOOLKIT=${TOOLKIT}
                              BUILD_TYPE=${EXATENSOR_BUILD_TYPE}
                              EXA_NO_BUILD=${TALSHXX_EXISTS}
                              EXA_TALSH_ONLY=NO
                              EXATN_SERVICE=YES
                              EXA_OS=LINUX
                              GPU_CUDA=CUDA
                              PATH_CUDA=${CUDA_TOOLKIT_ROOT_DIR}
                              CUDA_HOST_COMPILER=${CUDA_HOST_COMPILER}
                              MPILIB=${MPI_LIB}
                              PATH_${MPI_LIB}=${MPI_ROOT_DIR}
                              PATH_${MPI_LIB}_BIN=${MPI_BIN_PATH}
                              BLASLIB=${BLAS_LIB}
                              PATH_BLAS_${BLAS_LIB}=${BLAS_PATH}
                              PATH_INTEL=${PATH_INTEL_ROOT}
                              WITH_LAPACK=${WITH_LAPACK}
                              PATH_LAPACK_LIB=${BLAS_PATH}
                              LAPACK_LIBS="${LAPACK_LIBRARIES}"
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
                              TOOLKIT=${TOOLKIT}
                              BUILD_TYPE=${EXATENSOR_BUILD_TYPE}
                              EXA_NO_BUILD=${TALSHXX_EXISTS}
                              EXA_TALSH_ONLY=NO
                              EXATN_SERVICE=YES
                              EXA_OS=LINUX
                              GPU_CUDA=CUDA
                              PATH_CUDA=${CUDA_TOOLKIT_ROOT_DIR}
                              CUDA_HOST_COMPILER=${CUDA_HOST_COMPILER}
                              MPILIB=${MPI_LIB}
                              PATH_${MPI_LIB}=${MPI_ROOT_DIR}
                              PATH_${MPI_LIB}_BIN=${MPI_BIN_PATH}
                              BLASLIB=${BLAS_LIB}
                              PATH_BLAS_${BLAS_LIB}=${BLAS_PATH}
                              PATH_INTEL=${PATH_INTEL_ROOT}
                              WITH_LAPACK=${WITH_LAPACK}
                              PATH_LAPACK_LIB=${BLAS_PATH}
                              LAPACK_LIBS="${LAPACK_LIBRARIES}"
                              make
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ExaTensor)
    endif()

  else()
    message(STATUS "This is Linux build without GPU CUDA support")

    if(NOT MPI_LIB OR MPI_LIB STREQUAL "NONE")
      add_custom_target(exatensor-build
                        COMMAND ${CMAKE_COMMAND}
                              -E
                              env
                              CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                              CMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                              CMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
                              TOOLKIT=${TOOLKIT}
                              BUILD_TYPE=${EXATENSOR_BUILD_TYPE}
                              EXA_NO_BUILD=${TALSHXX_EXISTS}
                              EXA_TALSH_ONLY=YES
                              EXATN_SERVICE=YES
                              EXA_OS=LINUX
                              GPU_CUDA=NOCUDA
                              MPILIB=NONE
                              BLASLIB=${BLAS_LIB}
                              PATH_BLAS_${BLAS_LIB}=${BLAS_PATH}
                              PATH_INTEL=${PATH_INTEL_ROOT}
                              WITH_LAPACK=${WITH_LAPACK}
                              PATH_LAPACK_LIB=${BLAS_PATH}
                              LAPACK_LIBS="${LAPACK_LIBRARIES}"
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
                              TOOLKIT=${TOOLKIT}
                              BUILD_TYPE=${EXATENSOR_BUILD_TYPE}
                              EXA_NO_BUILD=${TALSHXX_EXISTS}
                              EXA_TALSH_ONLY=NO
                              EXATN_SERVICE=YES
                              EXA_OS=LINUX
                              GPU_CUDA=NOCUDA
                              MPILIB=${MPI_LIB}
                              PATH_${MPI_LIB}=${MPI_ROOT_DIR}
                              PATH_${MPI_LIB}_BIN=${MPI_BIN_PATH}
                              BLASLIB=${BLAS_LIB}
                              PATH_BLAS_${BLAS_LIB}=${BLAS_PATH}
                              PATH_INTEL=${PATH_INTEL_ROOT}
                              WITH_LAPACK=${WITH_LAPACK}
                              PATH_LAPACK_LIB=${BLAS_PATH}
                              LAPACK_LIBS="${LAPACK_LIBRARIES}"
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
                              TOOLKIT=${TOOLKIT}
                              BUILD_TYPE=${EXATENSOR_BUILD_TYPE}
                              EXA_NO_BUILD=${TALSHXX_EXISTS}
                              EXA_TALSH_ONLY=NO
                              EXATN_SERVICE=YES
                              EXA_OS=LINUX
                              GPU_CUDA=NOCUDA
                              MPILIB=${MPI_LIB}
                              PATH_${MPI_LIB}=${MPI_ROOT_DIR}
                              PATH_${MPI_LIB}_BIN=${MPI_BIN_PATH}
                              BLASLIB=${BLAS_LIB}
                              PATH_BLAS_${BLAS_LIB}=${BLAS_PATH}
                              PATH_INTEL=${PATH_INTEL_ROOT}
                              WITH_LAPACK=${WITH_LAPACK}
                              PATH_LAPACK_LIB=${BLAS_PATH}
                              LAPACK_LIBS="${LAPACK_LIBRARIES}"
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
        "${PROJECT_NAME} requires g++ 5.0 or greater. ExaTN will not build ANTLR support."
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
        "${PROJECT_NAME} requires clang 4.2.1 or greater. ExaTN will not build ANTLR support."
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
                   COMMENT "Installing ExaTENSOR Libs to Install Prefix...")
add_custom_command(
  TARGET exatensor-build POST_BUILD
  COMMAND ${CMAKE_COMMAND}
          -E
          copy
          ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include/tensor_method.hpp
          ${CMAKE_INSTALL_PREFIX}/include/tensor_method.hpp
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Installing ExaTENSOR tensor_method.hpp to Install Prefix...")
add_custom_command(
  TARGET exatensor-build POST_BUILD
  COMMAND ${CMAKE_COMMAND}
          -E
          copy
          ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include/talshxx.hpp
          ${CMAKE_INSTALL_PREFIX}/include/talshxx.hpp
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Installing ExaTENSOR talshxx.hpp to Install Prefix...")
add_custom_command(
  TARGET exatensor-build POST_BUILD
  COMMAND ${CMAKE_COMMAND}
          -E
          copy
          ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include/talsh.h
          ${CMAKE_INSTALL_PREFIX}/include/talsh.h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Installing ExaTENSOR talsh.h to Install Prefix...")
add_custom_command(
  TARGET exatensor-build POST_BUILD
  COMMAND ${CMAKE_COMMAND}
          -E
          copy
          ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include/timer.h
          ${CMAKE_INSTALL_PREFIX}/include/timer.h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Installing ExaTENSOR timer.h to Install Prefix...")
add_custom_command(
  TARGET exatensor-build POST_BUILD
  COMMAND ${CMAKE_COMMAND}
          -E
          copy
          ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include/tensor_algebra.h
          ${CMAKE_INSTALL_PREFIX}/include/tensor_algebra.h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Installing ExaTENSOR tensor_algebra.h to Install Prefix...")
add_custom_command(
  TARGET exatensor-build POST_BUILD
  COMMAND ${CMAKE_COMMAND}
          -E
          copy
          ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include/talsh_complex.h
          ${CMAKE_INSTALL_PREFIX}/include/talsh_complex.h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Installing ExaTENSOR talsh_complex.h to Install Prefix...")
add_custom_command(
  TARGET exatensor-build POST_BUILD
  COMMAND ${CMAKE_COMMAND}
          -E
          copy
          ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include/mem_manager.h
          ${CMAKE_INSTALL_PREFIX}/include/mem_manager.h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Installing ExaTENSOR mem_manager.h to Install Prefix...")
add_custom_command(
  TARGET exatensor-build POST_BUILD
  COMMAND ${CMAKE_COMMAND}
          -E
          copy
          ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include/talsh_task.hpp
          ${CMAKE_INSTALL_PREFIX}/include/talsh_task.hpp
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Installing ExaTENSOR talsh_task.hpp to Install Prefix...")
add_custom_command(
  TARGET exatensor-build POST_BUILD
  COMMAND ${CMAKE_COMMAND}
          -E
          copy
          ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include/byte_packet.h
          ${CMAKE_INSTALL_PREFIX}/include/byte_packet.h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Installing ExaTENSOR byte_packet.h to Install Prefix...")
