
add_subdirectory(util)

set(LIBRARY_NAME taprol-syntax-handler)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
add_library(${LIBRARY_NAME}
            SHARED
            taprol_syntax_handler.cpp)

target_include_directories(${LIBRARY_NAME}
                           PUBLIC util
                                  ${CLANG_INCLUDE_DIRS}
                                  ${LLVM_INCLUDE_DIRS})

target_link_libraries(${LIBRARY_NAME}
                      PRIVATE ${CLANG_LIBS} ${LLVM_LIBS} taprol-sh-util)

if(APPLE)
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "@loader_path/../lib;${LLVM_INSTALL_PREFIX}/lib")
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "$ORIGIN/../lib:${LLVM_INSTALL_PREFIX}/lib")
  set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-shared")
endif()

install(TARGETS ${LIBRARY_NAME} DESTINATION clang-plugins)
