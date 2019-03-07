set(LIBRARY_NAME exatn)
add_library(${LIBRARY_NAME}
            SHARED
            exatn.cpp
            ServiceRegistry.cpp
            )

target_include_directories(
  ${LIBRARY_NAME}
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
         ${CMAKE_SOURCE_DIR}/src/driver-rpc
         ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include
         ${CMAKE_SOURCE_DIR}/tpls/cppmicroservices/framework/include)

target_link_libraries(${LIBRARY_NAME} PUBLIC CppMicroServices)

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
