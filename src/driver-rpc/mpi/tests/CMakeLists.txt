add_executable(client_test client.cpp)
target_include_directories(client_test PRIVATE ${CUDA_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include ${MPI_CXX_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src/driver-rpc ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(client_test PUBLIC exatn ${MPI_CXX_LIBRARIES})

set_target_properties(client_test PROPERTIES FOLDER tests)
target_link_libraries(client_test PRIVATE gtest gmock gtest_main)

add_executable(server_test server.cpp)
target_include_directories(server_test PRIVATE ${CUDA_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include ${MPI_CXX_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src/driver-rpc ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(server_test PUBLIC exatn ${MPI_CXX_LIBRARIES} ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/lib/libtalsh.so ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvToolsExt.so ${BLAS_LIBRARIES} gfortran gomp)

add_test(NAME client_server_test COMMAND mpirun -np 1 ./server_test : -np 1 ./client_test)

add_dependencies(client_test exatensor-build)
add_dependencies(server_test exatensor-build)

add_subdirectory(example-tensor-method)