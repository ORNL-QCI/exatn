

if(MPI_FOUND)
  add_subdirectory(mpi)
endif()

file (GLOB HEADERS *.hpp)

install(FILES ${HEADERS} DESTINATION include/exatn)