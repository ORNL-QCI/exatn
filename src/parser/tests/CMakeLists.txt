exatn_add_mpi_test(TAProLInterpreterTester TAProLInterpreterTester.cpp)
target_include_directories(TAProLInterpreterTester
                           PUBLIC
                           ${CMAKE_SOURCE_DIR}/tpls/antlr/runtime/src
                           ..
                           ../generated
                          )

target_link_libraries(TAProLInterpreterTester PRIVATE
                      exatn-taprol-parser
                      exatn)
