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

if(CUDA_FOUND)
  message(STATUS "CUDA BLAS Library: ${CUDA_CUBLAS_LIBRARIES}")
  target_link_libraries(${LIBRARY_NAME} PUBLIC
                        exatn-utility
                        exatensor
                        talsh
                        blas
                        MPI::MPI_CXX
                        MPI::MPI_Fortran
                        ${CUDA_BLAS_LIB}
                        ${CUDA_LIBRARIES}
                        gfortran
                        gomp)
else()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNO_GPU")

  target_link_libraries(${LIBRARY_NAME} PUBLIC
                        exatn-utility
                        exatensor
                        talsh
                        blas
                        MPI::MPI_CXX
                        MPI::MPI_Fortran
                        gfortran
                        gomp)
endif()

if(EXATN_BUILD_TESTS)
  add_subdirectory(tests)
endif()
