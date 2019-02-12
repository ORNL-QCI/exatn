exatn_add_test(MPIRPCTester MPIRPCTester.cpp)
target_include_directories(MPIRPCTester PRIVATE testplugin ${CMAKE_SOURCE_DIR}/src/driver_rpc ${CMAKE_SOURCE_DIR}/src/driver_rpc/mpi ${CMAKE_BINARY_DIR})
target_link_libraries(MPIRPCTester PRIVATE
                        exatn-mpi-driver-rpc CppMicroServices ${MPI_CXX_LIBRARIES})