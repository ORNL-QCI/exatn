exatn_add_test(MPIRPCTester MPIRPCTester.cpp)
target_include_directories(MPIRPCTester PRIVATE ${CMAKE_SOURCE_DIR}/src/driver-rpc ${CMAKE_SOURCE_DIR}/src/driver-rpc/mpi ${CMAKE_BINARY_DIR})
target_link_libraries(MPIRPCTester PRIVATE
                        exatn-mpi-driver-rpc CppMicroServices ${MPI_CXX_LIBRARIES})

add_executable(client_test client.cpp)
target_include_directories(client_test PRIVATE ${CMAKE_SOURCE_DIR}/src/driver-rpc ${CMAKE_SOURCE_DIR}/src/driver-rpc/mpi)
target_link_libraries(client_test PRIVATE exatn-mpi-driver-rpc CppMicroServices ${MPI_CXX_LIBRARIES})

add_executable(server_test server.cpp)
target_include_directories(server_test PRIVATE ${CMAKE_SOURCE_DIR}/src/driver-rpc ${CMAKE_SOURCE_DIR}/src/driver-rpc/mpi)
target_link_libraries(server_test PRIVATE exatn-mpi-driver-rpc CppMicroServices ${MPI_CXX_LIBRARIES})