add_subdirectory(testplugin)
exatn_add_test(ServiceRegistryTester ServiceRegistryTester.cpp)
target_include_directories(ServiceRegistryTester PRIVATE testplugin ${CMAKE_SOURCE_DIR}/src/exatn ${CMAKE_BINARY_DIR})
target_link_libraries(ServiceRegistryTester PRIVATE exatn CppMicroServices)

exatn_add_test(NumServerTester NumServerTester.cpp)
#target_include_directories(NumServerTester PRIVATE testplugin ${CMAKE_SOURCE_DIR}/src/exatn ${CMAKE_BINARY_DIR})
target_link_libraries(NumServerTester PRIVATE exatn)
