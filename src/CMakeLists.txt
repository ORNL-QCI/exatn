add_subdirectory(utility)
add_subdirectory(numerics)

set(LIBRARY_NAME exatn)

add_library(${LIBRARY_NAME} SHARED exatn.cpp)

target_include_directories(
  ${LIBRARY_NAME}
  PUBLIC . utility ${CMAKE_SOURCE_DIR}/tpls/cppmicroservices/framework/include)

exatn_enable_rpath(${LIBRARY_NAME})

install(FILES ${HEADERS} DESTINATION include/exatn)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)