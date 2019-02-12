add_subdirectory(utility)
add_subdirectory(numerics)
add_subdirectory(driver_rpc)

set(LIBRARY_NAME exatn)

add_library(${LIBRARY_NAME} SHARED exatn.cpp)

target_include_directories(
  ${LIBRARY_NAME}
  PUBLIC . utility)

exatn_enable_rpath(${LIBRARY_NAME})

install(FILES ${HEADERS} DESTINATION include/exatn)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)