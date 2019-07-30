exatn_add_test(TAProLInterpreterTester TAProLInterpreterTester.cpp)
target_include_directories(TAProLInterpreterTester
                            PUBLIC
                            ${CMAKE_SOURCE_DIR}/tpls/antlr/runtime/src
                            ${CMAKE_SOURCE_DIR}/src/parser
                            ${CMAKE_SOURCE_DIR}/src/parser/generated
                            )

target_link_libraries(TAProLInterpreterTester PRIVATE
                      exatn-taprol-parser
                      exatn-numerics
                      exatn-runtime)