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
