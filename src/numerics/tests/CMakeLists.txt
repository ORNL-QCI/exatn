exatn_add_test(NumericsTester NumericsTester.cpp)
target_include_directories(NumericsTester PRIVATE ${CMAKE_SOURCE_DIR}/src/numerics)
target_link_directories(NumericsTester PRIVATE ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/lib)
if(CUDA_FOUND)
  target_link_libraries(NumericsTester PRIVATE
                        exatn-numerics
                        exatn-utility
                        talsh
                        blas
                        MPI::MPI_CXX
                        MPI::MPI_Fortran
                        ${CUDA_BLAS_LIB}
                        ${CUDA_LIBRARIES}
                        gfortran
                        gomp)
else()
  target_link_libraries(NumericsTester PRIVATE
                        exatn-numerics
                        exatn-utility
                        talsh
                        blas
                        MPI::MPI_CXX
                        MPI::MPI_Fortran
                        gfortran
                        gomp)
endif()