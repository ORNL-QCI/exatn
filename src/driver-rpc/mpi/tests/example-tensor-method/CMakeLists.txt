set(LIBRARY_NAME exatn-example-tensormethod)

file(GLOB_RECURSE HEADERS *.hpp)
file(GLOB SRC *.cpp)

# Set up dependencies to resources to track changes
usfunctiongetresourcesource(TARGET
                            ${LIBRARY_NAME}
                            OUT
                            SRC)
# Generate bundle initialization code
usfunctiongeneratebundleinit(TARGET
                             ${LIBRARY_NAME}
                             OUT
                             SRC)

add_library(${LIBRARY_NAME} SHARED ${SRC})

add_dependencies(${LIBRARY_NAME} exatensor-build)

if(CUDA_FOUND)
  target_include_directories(${LIBRARY_NAME}
                             PRIVATE ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include
                                     ${CMAKE_SOURCE_DIR}/src/exatn
                                     ${CUDA_INCLUDE_DIRS})

  target_link_libraries(
    ${LIBRARY_NAME}
    PUBLIC ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/lib/libtalsh.so
           ${CUDA_LIBRARIES}
           ${CUDA_CUBLAS_LIBRARIES}
           ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvToolsExt.so
           ${BLAS_LIBRARIES}
           gfortran
           gomp
           CppMicroServices)

else()
  target_include_directories(${LIBRARY_NAME}
                             PRIVATE ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include
                                     ${CMAKE_SOURCE_DIR}/src/exatn)

  if(APPLE)
    message(STATUS "Linking TALSH to example tensor method")
    target_link_libraries(
      ${LIBRARY_NAME}
      PUBLIC ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/lib/libtalsh.a
             ${BLAS_LIBRARIES}
             gfortran
             gomp
             CppMicroServices)
  else()
    target_link_libraries(
      ${LIBRARY_NAME}
      PUBLIC ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/lib/libtalsh.so
             ${BLAS_LIBRARIES}
             gfortran
             gomp
             CppMicroServices)
  endif()
endif()

set(_bundle_name exatn_example_tensormethod)

set_target_properties(${LIBRARY_NAME}
                      PROPERTIES # This is required for every bundle
                                 COMPILE_DEFINITIONS
                                 US_BUNDLE_NAME=${_bundle_name}
                                 # This is for convenience, used by other CMake
                                 # functions
                                 US_BUNDLE_NAME ${_bundle_name})

# Embed meta-data from a manifest.json file
usfunctionembedresources(TARGET
                         ${LIBRARY_NAME}
                         WORKING_DIRECTORY
                         ${CMAKE_CURRENT_SOURCE_DIR}
                         FILES
                         manifest.json)
exatn_configure_plugin_rpath(${LIBRARY_NAME})

install(TARGETS ${LIBRARY_NAME} DESTINATION plugins)
