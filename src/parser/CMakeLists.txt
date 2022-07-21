set (PACKAGE_NAME "TAProL Parser")
set (LIBRARY_NAME exatn-taprol-parser)

file (GLOB_RECURSE HEADERS *.hpp generated/*.hpp)
file (GLOB SRC *.cpp generated/*.cpp)

add_library(${LIBRARY_NAME} SHARED ${SRC})
target_include_directories(${LIBRARY_NAME}
                           PUBLIC . generated
                           ${CMAKE_SOURCE_DIR}/tpls/antlr/runtime/src
                          )


set(ANTLR_LIB ${CMAKE_SOURCE_DIR}/dist/libantlr4-runtime.so)
if (APPLE)
  set(ANTLR_LIB ${CMAKE_SOURCE_DIR}/dist/libantlr4-runtime.dylib)
endif()

add_dependencies(${LIBRARY_NAME} antlr4_shared)

target_link_libraries(${LIBRARY_NAME} PUBLIC exatn ${ANTLR_LIB})

exatn_configure_library_rpath(${LIBRARY_NAME})

if(EXATN_BUILD_TESTS)
  add_subdirectory(tests)
endif()

file (GLOB HEADERS *.hpp)

install(FILES ${HEADERS} DESTINATION include/exatn)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

if (NOT("${CMAKE_CXX_COMPILER_WRAPPER}" MATCHES "CrayPrgEnv"))
  find_package(Clang 10.0.0)
  # Only build Syntax Handler when using the custom Clang build.
  # CrayPrgEnv uses a Clang-based compiler.
  if (CLANG_FOUND) 
    add_subdirectory(syntax_handler)
  endif()
endif()