set(LIBRARY_NAME exatn-numerics-talsh)

file(GLOB SRC *.cpp)

usFunctionGetResourceSource(TARGET ${LIBRARY_NAME} OUT SRC)
usFunctionGenerateBundleInit(TARGET ${LIBRARY_NAME} OUT SRC)

add_library(${LIBRARY_NAME} SHARED ${SRC})

if(CUDA_FOUND)
  target_include_directories(${LIBRARY_NAME}
                             PUBLIC .
                                    ..
                                    ${CMAKE_SOURCE_DIR}/src
                                    ${CMAKE_SOURCE_DIR}/src/utility
                                    ${MPI_CXX_INCLUDE_DIRS}
                                    ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/TALSH
                                    ${CUDA_INCLUDE_DIRS})

else()
  target_include_directories(${LIBRARY_NAME}
                             PUBLIC .
                                    ..
                                    ${CMAKE_SOURCE_DIR}/src/utility
                                    ${MPI_CXX_INCLUDE_DIRS}
                                    ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/TALSH)

endif()

set(_bundle_name exatn_numerics_talsh)

set_target_properties(${LIBRARY_NAME}
                      PROPERTIES COMPILE_DEFINITIONS
                                 US_BUNDLE_NAME=${_bundle_name}
                                 US_BUNDLE_NAME
                                 ${_bundle_name})

usFunctionEmbedResources(TARGET
                         ${LIBRARY_NAME}
                         WORKING_DIRECTORY
                         ${CMAKE_CURRENT_SOURCE_DIR}
                         FILES
                         manifest.json)

target_link_libraries(${LIBRARY_NAME}
                      PUBLIC exatn exatn-utility ${MPI_CXX_LIBRARIES} CppMicroServices)
exatn_enable_rpath(${LIBRARY_NAME})

#if(EXATN_BUILD_TESTS)
#  add_subdirectory(tests)
#endif()

file(GLOB HEADERS *.hpp)

install(TARGETS ${LIBRARY_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/plugins)
