configure_file("${CMAKE_CURRENT_SOURCE_DIR}/exatn-config.cmake.in"
               "${CMAKE_BINARY_DIR}/exatn-config.cmake" @ONLY)
install(FILES "${CMAKE_BINARY_DIR}/exatn-config.cmake" DESTINATION .)


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/exatn-config.in
               ${CMAKE_BINARY_DIR}/scripts/exatn-config)
install(PROGRAMS ${CMAKE_BINARY_DIR}/scripts/exatn-config DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
