add_subdirectory(graph)
add_subdirectory(optimizer)
add_subdirectory(executor)

set(LIBRARY_NAME exatn-runtime)

file(GLOB SRC
     tensor_runtime.cpp
    )

add_library(${LIBRARY_NAME}
            SHARED
            ${SRC}
           )

add_dependencies(${LIBRARY_NAME} exatensor-build)

target_include_directories(${LIBRARY_NAME}
  PUBLIC . graph optimizer executor ${CMAKE_SOURCE_DIR}/src/exatn)

if(CUQUANTUM)
  target_include_directories(${LIBRARY_NAME} PUBLIC executor/cuquantum)
endif()

target_link_libraries(${LIBRARY_NAME}
                      PUBLIC CppMicroServices exatn-utils exatn-numerics exatn-runtime-graph
                      PRIVATE Boost::graph)

if(CUQUANTUM)
  target_link_libraries(${LIBRARY_NAME} PUBLIC exatn_runtime_cuquantum)
endif()

exatn_configure_library_rpath(${LIBRARY_NAME})

if(EXATN_BUILD_TESTS)
  add_subdirectory(tests)
endif()

file (GLOB HEADERS *.hpp)

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/exatn)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
