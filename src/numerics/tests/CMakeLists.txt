exatn_add_test(NumericsTester NumericsTester.cpp)
target_include_directories(NumericsTester
                           PRIVATE
                           exatn
                           ${CMAKE_SOURCE_DIR}/src/driver-rpc
                           ${CMAKE_SOURCE_DIR}/src/numerics
                          )

set(TALSH_LIB ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/lib/libtalsh.so)
if (APPLE)
  set(TALSH_LIB ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/lib/libtalsh.a)
endif()

if(CUDA_FOUND)
  link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  target_link_libraries(NumericsTester
                       PRIVATE
                        exatn
                        exatn-numerics
                        exatn-runtime
                        ${TALSH_LIB}
                        ${BLAS_LIBRARIES}
                        ${CUDA_LIBRARIES}
                        ${CUDA_CUBLAS_LIBRARIES}
                        ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvToolsExt.so
                        MPI::MPI_CXX
                        MPI::MPI_Fortran
                        gfortran
                        gomp)
else()
  target_link_libraries(NumericsTester
                       PRIVATE
                        exatn
                        exatn-numerics
                        exatn-runtime
                        ${TALSH_LIB}
                        ${BLAS_LIBRARIES}
                        MPI::MPI_CXX
                        MPI::MPI_Fortran
                        gfortran
                        gomp)
endif()
