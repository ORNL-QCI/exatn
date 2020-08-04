set(LIBRARY_NAME exatn-utils)

file(GLOB SRC
     mpi_proxy.cpp
    )

add_library(${LIBRARY_NAME}
            SHARED
            ${SRC}
           )

add_dependencies(${LIBRARY_NAME} exatensor-build)

target_include_directories(${LIBRARY_NAME} PUBLIC .)

target_link_libraries(${LIBRARY_NAME}
                      PUBLIC ExaTensor::ExaTensor)

exatn_configure_library_rpath(${LIBRARY_NAME})

file (GLOB HEADERS *.hpp)

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/exatn)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

