set(LIBRARY_NAME exatn-utility)
add_library(${LIBRARY_NAME} SHARED timer.cpp srep/Srep2TaProl.cpp)
target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
add_dependencies(${LIBRARY_NAME} exatensor-build)

if(EXATN_BUILD_TESTS)
  add_subdirectory(srep/tests)
endif()
