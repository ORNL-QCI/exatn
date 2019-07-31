# Create imported target CppMicroServices::CppMicroServices
add_library(ExaTensor::Talsh SHARED IMPORTED)

set(REQUIRED_TALSH_LIBS "${OpenMP_CXX_LIBRARIES};${OpenMP_Fortran_LIBRARIES}")
if (CUDA_FOUND)
  set(REQUIRED_TALSH_LIBS "${REQUIRED_TALSH_LIBS};${CUDA_LIBRARIES};${CUDA_CUBLAS_LIBRARIES}")
endif()

set_target_properties(ExaTensor::Talsh
                      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                 "${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include"
                                 INTERFACE_LINK_LIBRARIES
                                 "${REQUIRED_TALSH_LIBS}")

set_property(TARGET ExaTensor::Talsh
             APPEND
             PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ExaTensor::Talsh
                      PROPERTIES IMPORTED_LOCATION_RELEASE
                                 "${CMAKE_INSTALL_PREFIX}/lib/libtalsh.so"
                                 IMPORTED_SONAME_RELEASE
                                 "libtalsh.so")

add_subdirectory(exatn)
add_subdirectory(numerics)
add_subdirectory(driver-rpc)
add_subdirectory(parser)
add_subdirectory(runtime)
