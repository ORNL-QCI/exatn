set(LIBRARY_NAME exatn-runtime-graph)

file(GLOB SRC boost/DirectedBoostGraph.cpp GraphActivator.cpp)

usfunctiongetresourcesource(TARGET ${LIBRARY_NAME} OUT SRC)
usfunctiongeneratebundleinit(TARGET ${LIBRARY_NAME} OUT SRC)

add_library(${LIBRARY_NAME}
            SHARED
            ${SRC}
            )

target_include_directories(
  ${LIBRARY_NAME}
  PUBLIC . boost
         ${CMAKE_SOURCE_DIR}/src/exatn
         ${CMAKE_SOURCE_DIR}/tpls/cppmicroservices/framework/include
  )


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

target_link_libraries(${LIBRARY_NAME} PUBLIC CppMicroServices PRIVATE Boost::graph)

if(APPLE)
   set_target_properties(${LIBRARY_NAME} PROPERTIES INSTALL_RPATH "@loader_path/../lib")
   set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
   set_target_properties(${LIBRARY_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
   set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-shared")
endif()

if(EXATN_BUILD_TESTS)
  add_subdirectory(boost/tests)
endif()

file (GLOB HEADERS *.hpp boost/*.hpp)

install(FILES ${HEADERS} DESTINATION include/exatn/runtime)
install(TARGETS ${LIBRARY_NAME} DESTINATION plugins)

