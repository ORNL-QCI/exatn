exatn_add_test(ServiceRegistryTester ServiceRegistryTester.cpp)
target_include_directories(ServiceRegistryTester PRIVATE ${CMAKE_SOURCE_DIR}/src/utility)
target_link_libraries(ServiceRegistryTester PRIVATE
                        exatn-utility CppMicroServices)