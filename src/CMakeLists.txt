add_subdirectory(utility)
add_subdirectory(numerics)
add_subdirectory(driver-rpc)

set(LIBRARY_NAME exatn)

add_library(${LIBRARY_NAME} SHARED exatn.cpp)

add_dependencies(${LIBRARY_NAME} exatensor-build)

target_include_directories(
  ${LIBRARY_NAME}
  PUBLIC . utility driver-rpc ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include)

target_link_libraries(exatn PUBLIC CppMicroServices exatn-utility)

if(APPLE)
   set_target_properties(exatn PROPERTIES INSTALL_RPATH "@loader_path")
   set_target_properties(exatn PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
   set_target_properties(exatn PROPERTIES INSTALL_RPATH "$ORIGIN")
   set_target_properties(exatn PROPERTIES LINK_FLAGS "-shared")
endif()

file (GLOB HEADERS *.hpp)

install(FILES ${HEADERS} DESTINATION include/exatn)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)