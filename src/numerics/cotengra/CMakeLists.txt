
# Find Python
find_package(Python COMPONENTS Interpreter Development)

if(Python_FOUND)
  set(LIBRARY_NAME exatn-cotengra-optimizer)
  get_filename_component(PYTHON_LIB_NAME ${Python_LIBRARIES} NAME)
  configure_file(
    py-cotengra-optimizer.in.cpp
    ${CMAKE_BINARY_DIR}/src/numerics/cotengra/py-cotengra-optimizer.cpp)
  file(GLOB SRC
       ${CMAKE_BINARY_DIR}/src/numerics/cotengra/py-cotengra-optimizer.cpp)
  
  set(CMAKE_CXX_FLAGS
       "${CMAKE_CXX_FLAGS} -fno-strict-aliasing -O2 -g -pipe -Werror=format-security -Wp,-D_FORTIFY_SOURCE=2 -Wformat -fexceptions --param=ssp-buffer-size=4 -grecord-gcc-switches -m64 -mtune=native -D_GNU_SOURCE -fPIC -fwrapv"
  )
  
  usfunctiongetresourcesource(TARGET ${LIBRARY_NAME} OUT SRC)
  usfunctiongeneratebundleinit(TARGET ${LIBRARY_NAME} OUT SRC)

  add_library(${LIBRARY_NAME} SHARED ${SRC})

  target_include_directories(
    ${LIBRARY_NAME} PUBLIC . .. ${Python_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/tpls/pybind11/include
    ${CMAKE_SOURCE_DIR}/src/exatn)

  target_link_libraries(${LIBRARY_NAME} PRIVATE CppMicroServices Python::Python exatn-utils)

  set(_bundle_name exatn_cotengra_optimizer)
  set_target_properties(
    ${LIBRARY_NAME}
    PROPERTIES COMPILE_DEFINITIONS US_BUNDLE_NAME=${_bundle_name}
               US_BUNDLE_NAME ${_bundle_name})

  usfunctionembedresources(TARGET ${LIBRARY_NAME} WORKING_DIRECTORY
                           ${CMAKE_CURRENT_SOURCE_DIR} FILES manifest.json)
                    
  exatn_configure_plugin_rpath(${LIBRARY_NAME})
  install(TARGETS ${LIBRARY_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/plugins)

  if(EXATN_BUILD_TESTS)
    add_subdirectory(tests)
  endif()
else()
  message(STATUS "${BoldYellow} Python Interpreter or Libs not found, skipping Cotengra Plugin Build.${ColorReset}")
endif()