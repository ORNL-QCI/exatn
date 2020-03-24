include_directories(..)
exatn_add_test(ExaTN_PythonTester exatn_py_api_tester.cpp)
target_link_libraries(ExaTN_PythonTester PRIVATE exatn Python::Python)