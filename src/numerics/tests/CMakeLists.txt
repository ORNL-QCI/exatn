exatn_add_test(NumericsTester NumericsTester.cpp)
target_include_directories(NumericsTester PRIVATE ${CMAKE_SOURCE_DIR}/src/numerics)
target_link_directories(NumericsTester PRIVATE ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/lib)
if(CUDA_FOUND)
  link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  target_link_libraries(NumericsTester PRIVATE
                        exatn-numerics
                        ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/lib/libtalsh.so
                        blas
                        ${CUDA_LIBRARIES}
                        ${CUDA_CUBLAS_LIBRARIES}
                        ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvToolsExt.so
                        MPI::MPI_CXX
                        MPI::MPI_Fortran
                        gfortran
                        gomp)
else()
  target_link_libraries(NumericsTester PRIVATE
                        exatn-numerics
                        ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/lib/libtalsh.so
                        blas
                        MPI::MPI_CXX
                        MPI::MPI_Fortran
                        gfortran
                        gomp)
endif()