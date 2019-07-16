set(LIBRARY_NAME exatn-test-plugin)

file(GLOB_RECURSE HEADERS *.hpp)
file(GLOB SRC *.cpp)

# Set up dependencies to resources to track changes
usfunctiongetresourcesource(TARGET ${LIBRARY_NAME} OUT SRC)
# Generate bundle initialization code
usfunctiongeneratebundleinit(TARGET ${LIBRARY_NAME} OUT SRC)

add_library(${LIBRARY_NAME} SHARED ${SRC})
target_include_directories(
  ${LIBRARY_NAME}
  PRIVATE ${CMAKE_SOURCE_DIR}/src/exatn)
target_link_libraries(${LIBRARY_NAME} PUBLIC CppMicroServices)
set(_bundle_name exatn_test_plugin)

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
