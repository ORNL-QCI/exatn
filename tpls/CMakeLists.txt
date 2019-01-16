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

  set(BUILD_SHARED_LIBS TRUE)
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

list(GET MPI_Fortran_INCLUDE_PATH 0 MPI_FORTRAN_INCLUDE)
get_filename_component(MPI_ROOT_DIR ${MPI_FORTRAN_INCLUDE} DIRECTORY)
get_filename_component(MPI_BIN_PATH ${MPI_CXX_COMPILER} DIRECTORY)

message(STATUS "MPI bin path: ${MPI_BIN_PATH}")
message(STATUS "MPI root path: ${MPI_ROOT_DIR}")

if(CUDA_FOUND)
  message(STATUS "CUDA ROOT: ${CUDA_TOOLKIT_ROOT_DIR}")
  message(STATUS "CUDA LIBRARIES: ${CUDA_LIBRARIES}")
  message(STATUS "CUDA BLAS LIBRARIES: ${CUDA_CUBLAS_LIBRARIES}")
  
  if(CUDA_HOST_COMPILER)
    add_custom_target(
      exatensor-build
      COMMAND ${CMAKE_COMMAND} -E env GPU_CUDA=CUDA
              PATH_CUDA=${CUDA_TOOLKIT_ROOT_DIR} MPILIB=OPENMPI BLASLIB=ATLAS
              PATH_OPENMPI_INC=${MPI_ROOT_DIR}/include
              PATH_OPENMPI_LIB=${MPI_ROOT_DIR}/lib
              PATH_OPENMPI_BIN=${MPI_BIN_PATH} CUDA_HOST_COMPILER=${CUDA_HOST_COMPILER}
              make
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ExaTensor)
  else()
    add_custom_target(
      exatensor-build
      COMMAND ${CMAKE_COMMAND} -E env GPU_CUDA=CUDA
              PATH_CUDA=${CUDA_TOOLKIT_ROOT_DIR} MPILIB=OPENMPI BLASLIB=ATLAS
              PATH_OPENMPI_INC=${MPI_ROOT_DIR}/include
              PATH_OPENMPI_LIB=${MPI_ROOT_DIR}/lib
              PATH_OPENMPI_BIN=${MPI_BIN_PATH} make
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ExaTensor)
  endif()
else()
  add_custom_target(
    exatensor-build
    COMMAND ${CMAKE_COMMAND} -E env MPILIB=OPENMPI BLASLIB=ATLAS
            PATH_OPENMPI_INC=${MPI_ROOT_DIR}/include
            PATH_OPENMPI_LIB=${MPI_ROOT_DIR}/lib
            PATH_OPENMPI_BIN=${MPI_BIN_PATH} make
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ExaTensor)
endif()
