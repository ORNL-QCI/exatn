

if(NOT ${MPI_LIB} STREQUAL "NONE")
  add_subdirectory(mpi)
endif()

file (GLOB HEADERS *.hpp)

install(FILES ${HEADERS} DESTINATION include/exatn)