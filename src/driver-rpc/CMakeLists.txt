
add_subdirectory(mpi)

file (GLOB HEADERS *.hpp)

install(FILES ${HEADERS} DESTINATION include/exatn)