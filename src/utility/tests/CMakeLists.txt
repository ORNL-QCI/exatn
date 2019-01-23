add_subdirectory(testplugin)
exatn_add_test(ServiceRegistryTester ServiceRegistryTester.cpp)
target_include_directories(ServiceRegistryTester PRIVATE testplugin ${CMAKE_SOURCE_DIR}/src/utility ${CMAKE_BINARY_DIR})
target_link_libraries(ServiceRegistryTester PRIVATE
                        exatn-utility CppMicroServices)