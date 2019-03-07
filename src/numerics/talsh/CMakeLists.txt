set(LIBRARY_NAME exatn-numerics-talsh)

file(GLOB SRC *.cpp)

usfunctiongetresourcesource(TARGET ${LIBRARY_NAME} OUT SRC)
usfunctiongeneratebundleinit(TARGET ${LIBRARY_NAME} OUT SRC)

add_library(${LIBRARY_NAME} SHARED ${SRC})

add_dependencies(${LIBRARY_NAME} exatensor-build)

if(CUDA_FOUND)
  target_include_directories(${LIBRARY_NAME}
                             PUBLIC .
                                    ..
                                    ${CMAKE_SOURCE_DIR}/src
                                    ${CMAKE_SOURCE_DIR}/src/utility
                                    ${MPI_CXX_INCLUDE_DIRS}
                                    ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include
                                    ${CUDA_INCLUDE_DIRS})

  target_link_libraries(
    ${LIBRARY_NAME}
    PUBLIC exatn
           ${MPI_CXX_LIBRARIES}
           CppMicroServices
           ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/lib/libtalsh.so
           blas
           ${CUDA_LIBRARIES}
           ${CUDA_CUBLAS_LIBRARIES})

else()
  target_include_directories(${LIBRARY_NAME}
                             PUBLIC .
                                    ..
                                    ${CMAKE_SOURCE_DIR}/src/utility
                                    ${MPI_CXX_INCLUDE_DIRS}
                                    ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include)
  target_link_libraries(
    ${LIBRARY_NAME}
    PUBLIC exatn
           ${MPI_CXX_LIBRARIES}
           CppMicroServices
           ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/lib/libtalsh.so
           blas)
endif()

set(_bundle_name exatn_numerics_talsh)

set_target_properties(${LIBRARY_NAME}
                      PROPERTIES COMPILE_DEFINITIONS
                                 US_BUNDLE_NAME=${_bundle_name}
                                 US_BUNDLE_NAME
                                 ${_bundle_name})

usfunctionembedresources(TARGET
                         ${LIBRARY_NAME}
                         WORKING_DIRECTORY
                         ${CMAKE_CURRENT_SOURCE_DIR}
                         FILES
                         manifest.json)

exatn_enable_rpath(${LIBRARY_NAME})

# if(EXATN_BUILD_TESTS) add_subdirectory(tests) endif()

file(GLOB HEADERS *.hpp)

install(TARGETS ${LIBRARY_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/plugins)
