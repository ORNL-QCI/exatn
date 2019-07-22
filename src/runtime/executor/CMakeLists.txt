set(LIBRARY_NAME exatn-runtime-executor)

file(GLOB SRC GraphExecutorActivator.cpp exatensor/ExatensorExecutor.cpp talsh/TalshExecutor.cpp)

usfunctiongetresourcesource(TARGET ${LIBRARY_NAME} OUT SRC)
usfunctiongeneratebundleinit(TARGET ${LIBRARY_NAME} OUT SRC)

add_library(${LIBRARY_NAME}
            SHARED
            ${SRC}
            )

target_include_directories(
  ${LIBRARY_NAME}
  PUBLIC . talsh exatensor
         ${CMAKE_SOURCE_DIR}/src/exatn
         ${CMAKE_SOURCE_DIR}/tpls/cppmicroservices/framework/include
  )


set(_bundle_name exatn_runtime_executor)
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

target_link_libraries(${LIBRARY_NAME} PUBLIC CppMicroServices exatn PRIVATE exatn-runtime-graph)

if(APPLE)
   set_target_properties(${LIBRARY_NAME} PROPERTIES INSTALL_RPATH "@loader_path/../lib;@loader_path")
   set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
   set_target_properties(${LIBRARY_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib;$ORIGIN")
   set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-shared")
endif()

if(EXATN_BUILD_TESTS)
  #add_subdirectory(boost/tests)
endif()

file (GLOB HEADERS *.hpp)

install(FILES ${HEADERS} DESTINATION include/exatn/runtime)
install(TARGETS ${LIBRARY_NAME} DESTINATION plugins)

