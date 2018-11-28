set(LIBRARY_NAME exatn-utility)
add_library(${LIBRARY_NAME} SHARED timer.cpp)
target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
