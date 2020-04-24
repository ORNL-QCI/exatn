set(LIBRARY_NAME exatn)
add_library(${LIBRARY_NAME}
            SHARED
            exatn.cpp
            exatn_service.cpp
            ServiceRegistry.cpp
            num_server.cpp
            reconstructor.cpp
            optimizer.cpp
            eigensolver.cpp)

add_dependencies(${LIBRARY_NAME} exatensor-build)

target_include_directories(${LIBRARY_NAME} PUBLIC . ${CMAKE_BINARY_DIR})

target_link_libraries(${LIBRARY_NAME}
                      PUBLIC CppMicroServices exatn-numerics exatn-runtime)

exatn_configure_library_rpath(exatn)

if(EXATN_BUILD_TESTS)
  add_subdirectory(tests)
endif()

file(GLOB HEADERS *.hpp)

install(FILES ${HEADERS} DESTINATION include/exatn)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
