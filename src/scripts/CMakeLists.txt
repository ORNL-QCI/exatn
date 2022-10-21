# Version info
set(MAJOR_VERSION 1)
set(MINOR_VERSION 0)
set(PATCH_VERSION 0)
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE EXATN_BUILD_VERSION ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "ExaTN GIT hash: ${EXATN_BUILD_VERSION}")
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/exatn-config.cmake.in"
    "${CMAKE_BINARY_DIR}/exatn-config.cmake" @ONLY)
install(FILES "${CMAKE_BINARY_DIR}/exatn-config.cmake" DESTINATION .)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/exatn-config.in
    ${CMAKE_BINARY_DIR}/scripts/exatn-config)
install(PROGRAMS ${CMAKE_BINARY_DIR}/scripts/exatn-config DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
