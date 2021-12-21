set(LIBRARY_NAME exatn-runtime-executor)

file(GLOB SRC
     node_executors/talsh/node_executor_talsh.cpp
     node_executors/exatensor/node_executor_exatensor.cpp
     graph_executors/eager/graph_executor_eager.cpp
     graph_executors/lazy/graph_executor_lazy.cpp
     executor_activator.cpp)

usfunctiongetresourcesource(TARGET ${LIBRARY_NAME} OUT SRC)
usfunctiongeneratebundleinit(TARGET ${LIBRARY_NAME} OUT SRC)

add_library(${LIBRARY_NAME} SHARED ${SRC})

target_include_directories(${LIBRARY_NAME}
  PUBLIC . ..
  node_executors/talsh
  node_executors/exatensor
  graph_executors/eager
  graph_executors/lazy
  ../graph
  ${CMAKE_SOURCE_DIR}/src/exatn
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

target_link_libraries(${LIBRARY_NAME}
  PUBLIC CppMicroServices exatn-numerics exatn-runtime
  PRIVATE ExaTensor::ExaTensor
)

if(CUQUANTUM)
  target_link_libraries(${LIBRARY_NAME} PUBLIC exatn_runtime_cuquantum)
endif()

exatn_configure_plugin_rpath(${LIBRARY_NAME})

if(CUQUANTUM)
  add_subdirectory(cuquantum)
endif()

file (GLOB HEADERS *.hpp)

install(FILES ${HEADERS} DESTINATION include/exatn)
install(TARGETS ${LIBRARY_NAME} DESTINATION plugins)
