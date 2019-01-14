set(LIBRARY_NAME exatn-numerics)
add_library(${LIBRARY_NAME}
            SHARED
            tensor_signature.cpp
            tensor_shape.cpp
            tensor_leg.cpp
            tensor.cpp
            tensor_exa.cpp
            tensor_factory.cpp)
target_include_directories(${LIBRARY_NAME} PUBLIC .)
add_dependencies(${LIBRARY_NAME} exatensor-build)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNO_GPU")

if(CUDA_FOUND)
   target_link_libraries(${LIBRARY_NAME} exatn-utility exatensor talsh blas ${MPI_CXX_LIBRARIES} ${MPI_Fortran_LIBRARIES} gfortran gomp ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
else()
   target_link_libraries(${LIBRARY_NAME} exatn-utility exatensor talsh blas ${MPI_CXX_LIBRARIES} ${MPI_Fortran_LIBRARIES} gfortran gomp)
endif()

if(EXATN_BUILD_TESTS)
  add_subdirectory(tests)
endif()
