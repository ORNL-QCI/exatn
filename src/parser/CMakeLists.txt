set (PACKAGE_NAME "TAProL Parser")
set (LIBRARY_NAME exatn-taprol-parser)

file (GLOB_RECURSE HEADERS *.hpp generated/*.hpp)
file (GLOB SRC *.cpp generated/*.cpp)

add_library(${LIBRARY_NAME} SHARED ${SRC})
target_include_directories(${LIBRARY_NAME}
                            PUBLIC . generated
                            ${CMAKE_SOURCE_DIR}/tpls/antlr/runtime/src
                            )

add_dependencies(${LIBRARY_NAME} antlr4_shared)
if (APPLE)
  target_link_libraries(${LIBRARY_NAME} PUBLIC exatn ${CMAKE_SOURCE_DIR}/dist/libantlr4-runtime.dylib)
else()
  target_link_libraries(${LIBRARY_NAME} PUBLIC exatn ${CMAKE_SOURCE_DIR}/dist/libantlr4-runtime.so)
endif()

exatn_configure_library_rpath(${LIBRARY_NAME})

if(EXATN_BUILD_TESTS)
  add_subdirectory(tests)
endif()

file (GLOB HEADERS *.hpp)

install(FILES ${HEADERS} DESTINATION include/exatn)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
