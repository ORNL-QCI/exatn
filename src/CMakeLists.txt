add_library(ExaTensor::ExaTensor SHARED IMPORTED)

set(REQUIRED_EXATENSOR_LIBS "${OpenMP_CXX_LIBRARIES};${OpenMP_Fortran_LIBRARIES};gfortran")
set(REQUIRED_EXATENSOR_INCLUDES "${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include")

if(CUDA_FOUND)
  message(STATUS "CUDA ROOT DIR ${CUDA_TOOLKIT_ROOT_DIR}")
  message(STATUS "CUDA LIBS ${CUDA_LIBRARIES}")

  set(REQUIRED_EXATENSOR_LIBS "${REQUIRED_EXATENSOR_LIBS};${CUDA_LIBRARIES};${CUDA_CUBLAS_LIBRARIES};${CUDA_cudadevrt_LIBRARY};${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvToolsExt.so")
  set(REQUIRED_EXATENSOR_INCLUDES "${REQUIRED_EXATENSOR_INCLUDES};${CUDA_INCLUDE_DIRS}")

  if(CUTENSOR)
    if(CUTENSOR_PATH AND NOT CUTENSOR_PATH STREQUAL ".")
      message(STATUS "CUTENSOR ROOT DIR ${CUTENSOR_PATH}")
      set(REQUIRED_EXATENSOR_INCLUDES "${REQUIRED_EXATENSOR_INCLUDES};${CUTENSOR_PATH}/include")
      set(REQUIRED_EXATENSOR_LIBS "${CUTENSOR_PATH}/lib/libcutensor.so;${REQUIRED_EXATENSOR_LIBS}")
    else()
      message(STATUS "CUTENSOR IS IN SYSTEM DIRECTORIES")
      set(REQUIRED_EXATENSOR_LIBS "cutensor;${REQUIRED_EXATENSOR_LIBS}")
    endif()
    if(CUQUANTUM AND CUQUANTUM_PATH)
      message(STATUS "CUQUANTUM ROOT DIR ${CUQUANTUM_PATH}")
    endif()
  endif()
endif()

if(BLAS_LIB AND BLAS_PATH)

  set(BLAS_LIBRARIES "")

  if(BLAS_LIB STREQUAL "ATLAS")

    find_library(BLAS_LIBRARY blas HINTS ${BLAS_PATH})
    set(BLAS_LIBRARIES "${BLAS_LIBRARY}")

  elseif(BLAS_LIB STREQUAL "MKL")

    find_library(lp64 mkl_intel_lp64 HINTS ${BLAS_PATH})
    find_library(gnu_thread mkl_gnu_thread HINTS ${BLAS_PATH})
    find_library(core mkl_core HINTS ${BLAS_PATH})

    get_filename_component(EXATN_MKL_PATH ${gnu_thread} DIRECTORY)

    set(BLAS_LIBRARIES "${lp64};${gnu_thread};${core};pthread;m;dl")

  elseif(BLAS_LIB STREQUAL "OPENBLAS")

    find_library(openblas openblas HINTS ${BLAS_PATH})
    set(BLAS_LIBRARIES "${openblas}")

  elseif(BLAS_LIB STREQUAL "ACML")

    find_library(acml_mp acml_mp HINTS ${BLAS_PATH})
    set(BLAS_LIBRARIES "${acml_mp}")

  elseif(BLAS_LIB STREQUAL "ESSL")

    find_library(essl essl HINTS ${BLAS_PATH})
    find_library(xlf90_r xlf90_r HINTS ${PATH_IBM_XL_FOR})
    find_library(xlfmath xlfmath HINTS ${PATH_IBM_XL_FOR})
    set(BLAS_LIBRARIES "${essl};${xlf90_r};${xlfmath}")

  endif()

  message(STATUS "Appending ${BLAS_LIBRARIES} to ExaTensor Libraries")
  set(REQUIRED_EXATENSOR_LIBS "${BLAS_LIBRARIES};${REQUIRED_EXATENSOR_LIBS}")

endif()

if(WITH_LAPACK)

  set(LAPACK_LIBRARIES "")

  if(BLAS_LIB STREQUAL "ATLAS")

    find_library(LAPACK_LIBRARY lapack HINTS ${BLAS_PATH})
    set(LAPACK_LIBRARIES "${LAPACK_LIBRARY}")

  elseif(BLAS_LIB STREQUAL "MKL")

    find_library(lp64 mkl_intel_lp64 HINTS ${BLAS_PATH})
    find_library(gnu_thread mkl_gnu_thread HINTS ${BLAS_PATH})
    find_library(core mkl_core HINTS ${BLAS_PATH})
    set(LAPACK_LIBRARIES "${lp64};${gnu_thread};${core};pthread;m;dl")

  elseif(BLAS_LIB STREQUAL "OPENBLAS")

    find_library(LAPACK_LIBRARY openblas HINTS ${BLAS_PATH})
    set(LAPACK_LIBRARIES "${LAPACK_LIBRARY}")

  elseif(BLAS_LIB STREQUAL "ACML")

    find_library(acml_mp acml_mp HINTS ${BLAS_PATH})
    set(LAPACK_LIBRARIES "${acml_mp}")

  elseif(BLAS_LIB STREQUAL "ESSL")

    find_library(essl essl HINTS ${BLAS_PATH})
    find_library(xlf90_r xlf90_r HINTS ${PATH_IBM_XL_FOR})
    find_library(xlfmath xlfmath HINTS ${PATH_IBM_XL_FOR})
    set(LAPACK_LIBRARIES "${essl};${xlf90_r};${xlfmath}")

  endif()

  message(STATUS "Appending ${LAPACK_LIBRARIES} to ExaTensor Libraries")
  set(REQUIRED_EXATENSOR_LIBS "${REQUIRED_EXATENSOR_LIBS};${LAPACK_LIBRARIES}")
endif()

if(MPI_ROOT_DIR)
  set(REQUIRED_EXATENSOR_LIBS "${REQUIRED_EXATENSOR_LIBS};${MPI_CXX_LIBRARIES};${MPI_Fortran_LIBRARIES}")
  set(REQUIRED_EXATENSOR_INCLUDES "${REQUIRED_EXATENSOR_INCLUDES};${MPI_CXX_INCLUDE_DIRS};${MPI_Fortran_INCLUDE_DIRS}")
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
  set(FULLPATHLIBS "${CMAKE_INSTALL_PREFIX}/lib/libtalsh${CMAKE_SHARED_LIBRARY_SUFFIX}")
  set(NOTFULLPATHLIBS "libtalsh${CMAKE_SHARED_LIBRARY_SUFFIX}")

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
  set(FULLPATHLIBS "${CMAKE_INSTALL_PREFIX}/lib/libtalsh${CMAKE_SHARED_LIBRARY_SUFFIX};${CMAKE_INSTALL_PREFIX}/lib/libexatensor${CMAKE_SHARED_LIBRARY_SUFFIX}")
  set(NOTFULLPATHLIBS "libtalsh${CMAKE_SHARED_LIBRARY_SUFFIX};libexatensor${CMAKE_SHARED_LIBRARY_SUFFIX}")

  if(APPLE)
    set(FULLPATHLIBS "${CMAKE_INSTALL_PREFIX}/lib/libtalsh.a;${CMAKE_INSTALL_PREFIX}/lib/libexatensor.a")
    set(NOTFULLPATHLIBS "libtalsh.a;libexatensor.a")
  endif()

  set_target_properties(ExaTensor::ExaTensor
                        PROPERTIES IMPORTED_LOCATION_RELEASE
                                   ${FULLPATHLIBS}
                                   IMPORTED_SONAME_RELEASE
                                   ${NOTFULLPATHLIBS})

endif()

#GNU ASAN activation for debugging:
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND GNU_ASAN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer -Wno-maybe-uninitialized -Wno-unused-result")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
  set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
endif()

add_subdirectory(utils)
add_subdirectory(exatn)
add_subdirectory(numerics)
add_subdirectory(driver-rpc)
add_subdirectory(parser)
add_subdirectory(runtime)
add_subdirectory(scripts)
