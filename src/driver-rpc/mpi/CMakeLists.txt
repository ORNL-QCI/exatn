set(LIBRARY_NAME exatn-mpi-rpc)

file(GLOB SRC *.cpp)

usfunctiongetresourcesource(TARGET ${LIBRARY_NAME} OUT SRC)
usfunctiongeneratebundleinit(TARGET ${LIBRARY_NAME} OUT SRC)

add_library(${LIBRARY_NAME} SHARED ${SRC})

add_dependencies(${LIBRARY_NAME} exatensor-build)

if(CUDA_FOUND)
  target_include_directories(${LIBRARY_NAME}
                             PUBLIC .
                                    ..
                                    ${CMAKE_SOURCE_DIR}/src/utility
                                    ${MPI_CXX_INCLUDE_DIRS}
                                    ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include
                                    ${CUDA_INCLUDE_DIRS})
  target_link_libraries(
    ${LIBRARY_NAME}
    PUBLIC exatn
           exatn-numerics
           exatn-taprol-parser
           ${MPI_CXX_LIBRARIES}
           CppMicroServices
           ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/lib/libtalsh.so
           ${CUDA_LIBRARIES}
           ${CUDA_CUBLAS_LIBRARIES})

else()
  target_include_directories(${LIBRARY_NAME}
                             PUBLIC .
                                    ..
                                    ${CMAKE_SOURCE_DIR}/src/utility
                                    ${CMAKE_SOURCE_DIR}/src/numerics
                                    ${MPI_CXX_INCLUDE_DIRS}
                                    ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include)

  target_link_libraries(
    ${LIBRARY_NAME}
    PUBLIC exatn
           exatn-numerics
           exatn-taprol-parser
           ${MPI_CXX_LIBRARIES}
           CppMicroServices
           ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/lib/libtalsh.so)

endif()

set(_bundle_name exatn_mpi_rpc)
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

if(EXATN_BUILD_TESTS)
  add_subdirectory(tests)
endif()

file(GLOB HEADERS *.hpp)

install(TARGETS ${LIBRARY_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/plugins)
