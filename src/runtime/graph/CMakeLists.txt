set(LIBRARY_NAME exatn-runtime-graph)

file(GLOB SRC
     tensor_exec_state.cpp
     boost/directed_boost_graph.cpp
     graph_activator.cpp
    )

usfunctiongetresourcesource(TARGET ${LIBRARY_NAME} OUT SRC)
usfunctiongeneratebundleinit(TARGET ${LIBRARY_NAME} OUT SRC)

add_library(${LIBRARY_NAME}
            SHARED
            ${SRC}
           )

target_include_directories(${LIBRARY_NAME}
  PUBLIC . .. ${CMAKE_SOURCE_DIR}/src/exatn boost)

set(_bundle_name exatn_runtime_graph)
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

target_link_libraries(${LIBRARY_NAME}
                      PUBLIC CppMicroServices exatn-numerics
                      PRIVATE Boost::graph)

exatn_configure_plugin_rpath(${LIBRARY_NAME})

if(EXATN_BUILD_TESTS)
  add_subdirectory(boost/tests)
endif()

file (GLOB HEADERS *.hpp boost/*.hpp)

install(FILES ${HEADERS} DESTINATION include/exatn/runtime)
install(TARGETS ${LIBRARY_NAME} DESTINATION plugins)
