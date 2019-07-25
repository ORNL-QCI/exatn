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

if(MPI_FOUND)
  list(GET MPI_Fortran_INCLUDE_PATH 0 MPI_FORTRAN_INCLUDE)
  get_filename_component(MPI_ROOT_DIR ${MPI_FORTRAN_INCLUDE} DIRECTORY)
  get_filename_component(MPI_BIN_PATH ${MPI_CXX_COMPILER} DIRECTORY)
endif()

message(STATUS "MPI bin path: ${MPI_BIN_PATH}")
message(STATUS "MPI root path: ${MPI_ROOT_DIR}")
message(STATUS "CUDA Host Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "The MPI_LIB implementation is set to ${MPI_LIB}")
message(STATUS "The BLAS_LIB implementation is set to ${BLAS_LIB}")

if (APPLE)
  message(STATUS "THIS IS APPLE, BUILDING EXATENSOR ${CMAKE_Fortran_COMPILER}")

  add_custom_target(
    exatensor-build
    COMMAND ${CMAKE_COMMAND} -E env CPP_GNU=${CMAKE_CXX_COMPILER} CC_GNU=${CMAKE_C_COMPILER} FC_GNU=${CMAKE_Fortran_COMPILER} MPILIB=${MPI_LIB} BLASLIB=${BLAS_LIB}
            PATH_MPICH_INC=${MPI_ROOT_DIR}/include EXA_NO_BUILD=${TALSHXX_EXISTS}
            PATH_MPICH_LIB=${MPI_ROOT_DIR}/lib
            PATH_MPICH_BIN=${MPI_BIN_PATH} EXA_OS=NO_LINUX GPU_CUDA=NOCUDA EXA_TALSH_ONLY=YES EXATN_SERVICE=YES make
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ExaTensor)

else()

  if(CUDA_FOUND)
    message(STATUS "CUDA ROOT: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA LIBRARIES: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA BLAS LIBRARIES: ${CUDA_CUBLAS_LIBRARIES}")


    if(NOT CUDA_HOST_COMPILER)
      set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    endif()

    add_custom_target(
       exatensor-build
       COMMAND ${CMAKE_COMMAND} -E env CPP_GNU=${CMAKE_CXX_COMPILER} CC_GNU=${CMAKE_C_COMPILER} FC_GNU=${CMAKE_Fortran_COMPILER} GPU_CUDA=CUDA
                PATH_CUDA=${CUDA_TOOLKIT_ROOT_DIR} MPILIB=${MPI_LIB} BLASLIB=${BLAS_LIB}
                PATH_OPENMPI_INC=${MPI_ROOT_DIR}/include EXA_NO_BUILD=${TALSHXX_EXISTS}
                PATH_OPENMPI_LIB=${MPI_ROOT_DIR}/lib
                PATH_OPENMPI_BIN=${MPI_BIN_PATH} CUDA_HOST_COMPILER=${CUDA_HOST_COMPILER} EXATN_SERVICE=YES
                make
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ExaTensor)

  else()
      add_custom_target(
      exatensor-build
      COMMAND ${CMAKE_COMMAND} -E env CPP_GNU=${CMAKE_CXX_COMPILER} CC_GNU=${CMAKE_C_COMPILER} FC_GNU=${CMAKE_Fortran_COMPILER} MPILIB=${MPI_LIB} BLASLIB=${BLAS_LIB}
              PATH_OPENMPI_INC=${MPI_ROOT_DIR}/include EXA_NO_BUILD=${TALSHXX_EXISTS}
              PATH_OPENMPI_LIB=${MPI_ROOT_DIR}/lib
              PATH_OPENMPI_BIN=${MPI_BIN_PATH} EXATN_SERVICE=YES make
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ExaTensor)
  endif()
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


add_custom_command(TARGET exatensor-build POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_INSTALL_PREFIX}/lib
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/lib/lib* ${CMAKE_INSTALL_PREFIX}/lib/
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Installing ExaTensor Libs to Install Prefix..."
)
add_custom_command(TARGET exatensor-build POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include/tensor_method.hpp ${CMAKE_INSTALL_PREFIX}/include/exatensor
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Installing ExaTensor tensor_method.hpp to Install Prefix..."
)
