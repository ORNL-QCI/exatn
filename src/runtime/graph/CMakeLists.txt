set(LIBRARY_NAME exatn-runtime-graph)

file(GLOB SRC
     tensor_exec_state.cpp
    )

add_library(${LIBRARY_NAME}
            SHARED
            ${SRC}
           )

target_include_directories(${LIBRARY_NAME}
  PUBLIC . .. ${CMAKE_SOURCE_DIR}/src/exatn)


target_link_libraries(${LIBRARY_NAME}
                      PUBLIC CppMicroServices exatn-numerics)

exatn_configure_library_rpath(${LIBRARY_NAME})

add_subdirectory(boost)

file (GLOB HEADERS *.hpp)

install(FILES ${HEADERS} DESTINATION include/exatn)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
