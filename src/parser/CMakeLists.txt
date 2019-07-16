set (PACKAGE_NAME "TAProL Parser")
set (LIBRARY_NAME exatn-taprol-parser)

file (GLOB_RECURSE HEADERS *.hpp generated/*.hpp)
file (GLOB SRC *.cpp generated/*.cpp)

add_library(${LIBRARY_NAME} SHARED ${SRC})
target_include_directories(${LIBRARY_NAME}
                            PUBLIC
                            ${CMAKE_SOURCE_DIR}/tpls/antlr/runtime/src
                            ${CMAKE_CURRENT_SOURCE_DIR}
                            ${CMAKE_CURRENT_SOURCE_DIR}/generated
                            )

add_dependencies(${LIBRARY_NAME} antlr4_shared)
if (APPLE)
target_link_libraries(${LIBRARY_NAME} PUBLIC exatn exatn-numerics ${CMAKE_SOURCE_DIR}/dist/libantlr4-runtime.dylib)
else()
target_link_libraries(${LIBRARY_NAME} PUBLIC exatn exatn-numerics ${CMAKE_SOURCE_DIR}/dist/libantlr4-runtime.so)
endif()

if(APPLE)
   set_target_properties(${LIBRARY_NAME} PROPERTIES INSTALL_RPATH "@loader_path")
   set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
   set_target_properties(${LIBRARY_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN")
   set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-shared")
endif()

if(EXATN_BUILD_TESTS)
  add_subdirectory(tests)
endif()

file (GLOB HEADERS *.hpp)

install(FILES ${HEADERS} DESTINATION include/exatn)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
