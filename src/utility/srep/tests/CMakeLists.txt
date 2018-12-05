exatn_add_test(Srep2TaProlTester Srep2TaProlTester.cpp)
target_include_directories(Srep2TaProlTester PRIVATE ${CMAKE_SOURCE_DIR}/src/utility/srep)
target_link_libraries(Srep2TaProlTester PRIVATE exatn-utility)

