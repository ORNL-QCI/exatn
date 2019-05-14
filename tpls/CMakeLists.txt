set(BUILD_SHARED_LIBS TRUE)
include_directories(${CMAKE_BINARY_DIR}/tpls/cppmicroservices/include)
include_directories(${CMAKE_BINARY_DIR}/tpls/cppmicroservices/framework/include)
add_subdirectory(cppmicroservices)

find_package(Git QUIET)

if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
  execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
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
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ExaTensor/include/talshxx.hpp")
  set(TALSHXX_EXISTS "YES")
endif()

list(GET MPI_Fortran_INCLUDE_PATH 0 MPI_FORTRAN_INCLUDE)
get_filename_component(MPI_ROOT_DIR ${MPI_FORTRAN_INCLUDE} DIRECTORY)
get_filename_component(MPI_BIN_PATH ${MPI_CXX_COMPILER} DIRECTORY)

message(STATUS "MPI bin path: ${MPI_BIN_PATH}")
message(STATUS "MPI root path: ${MPI_ROOT_DIR}")
message(STATUS "CUDA Host Compiler: ${CMAKE_CXX_COMPILER}")

if(CUDA_FOUND)
  message(STATUS "CUDA ROOT: ${CUDA_TOOLKIT_ROOT_DIR}")
  message(STATUS "CUDA LIBRARIES: ${CUDA_LIBRARIES}")
  message(STATUS "CUDA BLAS LIBRARIES: ${CUDA_CUBLAS_LIBRARIES}")

  if(CUDA_HOST_COMPILER)
    add_custom_target(
      exatensor-build
      COMMAND ${CMAKE_COMMAND} -E env CPP_GNU=g++-8 CC_GNU=gcc-8 FC_GNU=gfortran-8 GPU_CUDA=CUDA
              PATH_CUDA=${CUDA_TOOLKIT_ROOT_DIR} MPILIB=OPENMPI BLASLIB=ATLAS
              PATH_OPENMPI_INC=${MPI_ROOT_DIR}/include EXA_NO_BUILD=${TALSHXX_EXISTS}
              PATH_OPENMPI_LIB=${MPI_ROOT_DIR}/lib
              PATH_OPENMPI_BIN=${MPI_BIN_PATH} CUDA_HOST_COMPILER=${CUDA_HOST_COMPILER}
              make
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ExaTensor)
  else()
    add_custom_target(
      exatensor-build
      COMMAND ${CMAKE_COMMAND} -E env CPP_GNU=g++-8 CC_GNU=gcc-8 FC_GNU=gfortran-8 GPU_CUDA=CUDA
              PATH_CUDA=${CUDA_TOOLKIT_ROOT_DIR} MPILIB=OPENMPI BLASLIB=ATLAS
              PATH_OPENMPI_INC=${MPI_ROOT_DIR}/include EXA_NO_BUILD=${TALSHXX_EXISTS}
              PATH_OPENMPI_LIB=${MPI_ROOT_DIR}/lib
              PATH_OPENMPI_BIN=${MPI_BIN_PATH} CUDA_HOST_COMPILER=${CMAKE_CXX_COMPILER} make
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ExaTensor)
  endif()
else()
  add_custom_target(
    exatensor-build
    COMMAND ${CMAKE_COMMAND} -E env CPP_GNU=g++-8 CC_GNU=gcc-8 FC_GNU=gfortran-8 MPILIB=OPENMPI BLASLIB=ATLAS
            PATH_OPENMPI_INC=${MPI_ROOT_DIR}/include EXA_NO_BUILD=${TALSHXX_EXISTS}
            PATH_OPENMPI_LIB=${MPI_ROOT_DIR}/lib
            PATH_OPENMPI_BIN=${MPI_BIN_PATH} make
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ExaTensor)
endif()

# Compiler-specific C++11 activation (FROM ANTLR4).
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
    # Just g++-5.0 and greater contain <codecvt> header. (test in ubuntu)
    if (NOT (GCC_VERSION VERSION_GREATER 5.0 OR GCC_VERSION VERSION_EQUAL 5.0))
        message(WARNING "${PROJECT_NAME} requires g++ 5.0 or greater. XACC will not build Antlr support.")
    else ()
        add_subdirectory(antlr)
    endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    add_subdirectory(antlr)
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND CMAKE_SYSTEM_NAME MATCHES "Linux")
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE CLANG_VERSION)
    if (NOT (CLANG_VERSION VERSION_GREATER 4.2.1 OR CLANG_VERSION VERSION_EQUAL 4.2.1))
        message(WARNING "${PROJECT_NAME} requires clang 4.2.1 or greater. XACC will not build Antlr support.")
    else ()
        add_subdirectory(antlr)
    endif()
    # You can use libc++ to compile this project when g++ is NOT greater than or equal to 5.0.
    if (WITH_LIBCXX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    endif()
endif ()

set(BUILD_SHARED_LIBS FALSE)
set(BOOST_LIBS_OPTIONAL graph CACHE STRING "" FORCE)
add_subdirectory(boost-cmake)