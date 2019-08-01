
exatn_add_test(DirectedBoostGraphTester DirectedBoostGraphTester.cpp)
target_include_directories(DirectedBoostGraphTester PRIVATE ${CMAKE_SOURCE_DIR}/src/runtime/graph/boost ${CMAKE_SOURCE_DIR}/src/runtime/graph ${CMAKE_SOURCE_DIR}/src/exatn ${CMAKE_SOURCE_DIR}/tpls/mpark-variant)
target_link_libraries(DirectedBoostGraphTester PRIVATE exatn Boost::graph)
