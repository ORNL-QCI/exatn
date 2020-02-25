set(LIBRARY_NAME exatn-mpi-rpc)

file(GLOB SRC *.cpp)

usfunctiongetresourcesource(TARGET ${LIBRARY_NAME} OUT SRC)
usfunctiongeneratebundleinit(TARGET ${LIBRARY_NAME} OUT SRC)

add_library(${LIBRARY_NAME} SHARED ${SRC})

add_dependencies(${LIBRARY_NAME} exatensor-build)

target_include_directories(${LIBRARY_NAME}
                             PUBLIC .
                                    ..
                                    ${MPI_CXX_INCLUDE_DIRS})

target_link_libraries(
    ${LIBRARY_NAME}
    PUBLIC exatn
           exatn-taprol-parser
           MPI::MPI_CXX)

set(_bundle_name exatn_mpi_rpc)
set_target_properties(${LIBRARY_NAME}
                      PROPERTIES COMPILE_DEFINITIONS
                                 US_BUNDLE_NAME=${_bundle_name}
                                 US_BUNDLE_NAME
                                 ${_bundle_name})

usfunctionembedresources(TARGET
                         ${LIBRARY_NAME}
                         WORKING_DIRECTORY
                         ${CMAKE_CURRENT_SOURCE_DIR}
                         FILES
                         manifest.json)

exatn_configure_plugin_rpath(${LIBRARY_NAME})

if(EXATN_BUILD_TESTS)
  add_subdirectory(tests)
endif()

file(GLOB HEADERS *.hpp)

install(TARGETS ${LIBRARY_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/plugins)
