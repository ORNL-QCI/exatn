set (PACKAGE_NAME "TAProL Parser")
set (LIBRARY_NAME exatn-taprol-parser)

file (GLOB_RECURSE HEADERS *.hpp generated/*.hpp)
file (GLOB SRC *.cpp generated/*.cpp)

include_directories(${CMAKE_SOURCE_DIR}/tpls/antlr/runtime/src)

include_directories(generated)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${CMAKE_SOURCE_DIR}/dist)

add_library(${LIBRARY_NAME} SHARED ${SRC})

target_link_libraries(${LIBRARY_NAME} PUBLIC exatn antlr4-runtime)

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
