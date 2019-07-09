add_subdirectory(graph)
add_subdirectory(optimizer)
add_subdirectory(executor)

set(LIBRARY_NAME exatn-runtime)

file(GLOB SRC *.cpp)

add_library(${LIBRARY_NAME}
            SHARED
            ${SRC}
            )

target_include_directories(
  ${LIBRARY_NAME}
  PUBLIC .
         boost
         ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include
         ${CMAKE_SOURCE_DIR}/src/exatn
         ${CMAKE_SOURCE_DIR}/tpls/cppmicroservices/framework/include
  PRIVATE
         ${CMAKE_SOURCE_DIR}/src/driver-rpc
  )

target_link_libraries(${LIBRARY_NAME} PUBLIC CppMicroServices exatn-numerics exatn-runtime-graph PRIVATE Boost::graph)

if(APPLE)
   set_target_properties(${LIBRARY_NAME} PROPERTIES INSTALL_RPATH "@loader_path")
   set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
   set_target_properties(${LIBRARY_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN")
   set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-shared")
endif()

if(EXATN_BUILD_TESTS)
  #add_subdirectory(tests)
endif()

file (GLOB HEADERS *.hpp boost/*.hpp)

install(FILES ${HEADERS} DESTINATION include/exatn/runtime)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
