set(LIBRARY_NAME exatn-example-tensormethod)

file(GLOB_RECURSE HEADERS *.hpp)
file(GLOB SRC *.cpp)

# Set up dependencies to resources to track changes
usFunctionGetResourceSource(TARGET ${LIBRARY_NAME} OUT SRC)
# Generate bundle initialization code
usFunctionGenerateBundleInit(TARGET ${LIBRARY_NAME} OUT SRC)

add_library(${LIBRARY_NAME} SHARED ${SRC})

add_dependencies(${LIBRARY_NAME} exatensor-build)

target_include_directories(
  ${LIBRARY_NAME}
  PRIVATE ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include ${CMAKE_SOURCE_DIR}/src/utility ${CUDA_INCLUDE_DIRS})

set(_bundle_name exatn_example_tensormethod)

set_target_properties(${LIBRARY_NAME}
                      PROPERTIES # This is required for every bundle
                                 COMPILE_DEFINITIONS
                                 US_BUNDLE_NAME=${_bundle_name}
                                 # This is for convenience, used by other CMake
                                 # functions
                                 US_BUNDLE_NAME ${_bundle_name})

# Embed meta-data from a manifest.json file
usFunctionEmbedResources(TARGET
                         ${LIBRARY_NAME}
                         WORKING_DIRECTORY
                         ${CMAKE_CURRENT_SOURCE_DIR}
                         FILES
                         manifest.json)

install(TARGETS ${LIBRARY_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/plugins)
