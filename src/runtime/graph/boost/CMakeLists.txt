set(LIBRARY_NAME exatn-runtime-boost-graph)

file(GLOB SRC
     directed_boost_graph.cpp
     boost_graph_activator.cpp
    )

usfunctiongetresourcesource(TARGET ${LIBRARY_NAME} OUT SRC)
usfunctiongeneratebundleinit(TARGET ${LIBRARY_NAME} OUT SRC)

add_library(${LIBRARY_NAME}
            SHARED
            ${SRC}
           )

target_include_directories(${LIBRARY_NAME}
  PUBLIC . ..)

set(_bundle_name exatn_runtime_boost_graph)
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
                      PUBLIC CppMicroServices exatn-runtime-graph
                      PRIVATE Boost::graph)

exatn_configure_plugin_rpath(${LIBRARY_NAME})

if(EXATN_BUILD_TESTS)
  add_subdirectory(tests)
endif()

install(TARGETS ${LIBRARY_NAME} DESTINATION plugins)
