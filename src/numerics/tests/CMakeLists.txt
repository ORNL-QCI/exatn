exatn_add_test(NumericsTester NumericsTester.cpp)
target_include_directories(NumericsTester PRIVATE ${CMAKE_SOURCE_DIR}/src/numerics)
target_link_libraries(NumericsTester PRIVATE exatn-numerics)

