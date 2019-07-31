add_library(ExaTensor::ExaTensor SHARED IMPORTED)

set(REQUIRED_EXATENSOR_LIBS
    "${OpenMP_CXX_LIBRARIES};${OpenMP_Fortran_LIBRARIES}")
set(REQUIRED_EXATENSOR_INCLUDES "${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include")

if(CUDA_FOUND)
  set(REQUIRED_EXATENSOR_LIBS
      "${REQUIRED_EXATENSOR_LIBS};${CUDA_LIBRARIES};${CUDA_CUBLAS_LIBRARIES};${CUDA_cudadevrt_LIBRARY}")
  set(REQUIRED_EXATENSOR_INCLUDES "${REQUIRED_EXATENSOR_INCLUDES};${CUDA_INCLUDE_DIRS}")
endif()

if (BLAS_FOUND)
   set(REQUIRED_EXATENSOR_LIBS
      "${REQUIRED_EXATENSOR_LIBS};${BLAS_LIBRARIES}")
endif()

if(MPI_ROOT_DIR)
  set(
    REQUIRED_EXATENSOR_LIBS
    "${REQUIRED_EXATENSOR_LIBS};${MPI_CXX_LIBRARIES};${MPI_Fortran_LIBRARIES}")
  set(
    REQUIRED_EXATENSOR_INCLUDES
    "${REQUIRED_EXATENSOR_INCLUDES};${MPI_CXX_INCLUDE_DIRS};${MPI_Fortran_INCLUDE_DIRS}"
    )
endif()

set_target_properties(ExaTensor::ExaTensor
                      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                 "${REQUIRED_EXATENSOR_INCLUDES}"
                                 INTERFACE_LINK_LIBRARIES
                                 "${REQUIRED_EXATENSOR_LIBS}")

set_property(TARGET ExaTensor::ExaTensor
             APPEND
             PROPERTY IMPORTED_CONFIGURATIONS RELEASE)

if(EXA_TALSH_ONLY)
  set(FULLPATHLIBS "${CMAKE_INSTALL_PREFIX}/lib/libtalsh.so")
  set(NOTFULLPATHLIBS "libtalsh.so")

  if(APPLE)
    set(FULLPATHLIBS "${CMAKE_INSTALL_PREFIX}/lib/libtalsh.a")
    set(NOTFULLPATHLIBS "libtalsh.a")
  endif()

  set_target_properties(ExaTensor::ExaTensor
                        PROPERTIES IMPORTED_LOCATION_RELEASE
                                   ${FULLPATHLIBS}
                                   IMPORTED_SONAME_RELEASE
                                   ${NOTFULLPATHLIBS})

else()
  set(
    FULLPATHLIBS
    "${CMAKE_INSTALL_PREFIX}/lib/libtalsh.so;${CMAKE_INSTALL_PREFIX}/lib/libexatensor.so"
    )
  set(NOTFULLPATHLIBS "libtalsh.so;libexatensor.so")

  if(APPLE)
    set(
      FULLPATHLIBS
      "${CMAKE_INSTALL_PREFIX}/lib/libtalsh.a;${CMAKE_INSTALL_PREFIX}/lib/libexatensor.a"
      )
    set(NOTFULLPATHLIBS "libtalsh.a;libexatensor.a")
  endif()

  set_target_properties(ExaTensor::ExaTensor
                        PROPERTIES IMPORTED_LOCATION_RELEASE
                                   ${FULLPATHLIBS}
                                   IMPORTED_SONAME_RELEASE
                                   ${NOTFULLPATHLIBS})

endif()

add_subdirectory(exatn)
add_subdirectory(numerics)
add_subdirectory(driver-rpc)
add_subdirectory(parser)
add_subdirectory(runtime)
