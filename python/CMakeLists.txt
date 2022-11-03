include_directories(${CMAKE_SOURCE_DIR}/src/exatn)
include_directories(${CMAKE_SOURCE_DIR}/src/driver-rpc)
include_directories(${CMAKE_SOURCE_DIR}/src/numerics)
include_directories(${Python_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/tpls/pybind11/include)
include_directories(${CMAKE_SOURCE_DIR}/tpls/ExaTensor/TALSH)
include_directories(${CMAKE_SOURCE_DIR}/tpls/eigen)

if (CUDA_FOUND)
  include_directories(${CUDA_INCLUDE_DIRS})
endif()

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64") 
   set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing -O2 -g -pipe -Werror=format-security -Wp,-D_FORTIFY_SOURCE=2 -Wformat -fexceptions --param=ssp-buffer-size=4 -grecord-gcc-switches -D_GNU_SOURCE -fPIC -fwrapv")
else()
   set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing -O2 -g -pipe -Werror=format-security -Wp,-D_FORTIFY_SOURCE=2 -Wformat -fexceptions --param=ssp-buffer-size=4 -grecord-gcc-switches -m64 -D_GNU_SOURCE -fPIC -fwrapv")
endif()

if(APPLE)
   set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif(APPLE)

add_library(_pyexatn SHARED exatn-py.cpp)

set_target_properties(_pyexatn PROPERTIES PREFIX "")

target_link_libraries(_pyexatn PUBLIC CppMicroServices exatn exatn-numerics)

if(APPLE)
   set_target_properties(_pyexatn PROPERTIES INSTALL_RPATH "@loader_path/lib;@loader_path/plugins")
   set_target_properties(_pyexatn PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
   set_target_properties(_pyexatn PROPERTIES INSTALL_RPATH "$ORIGIN/lib;$ORIGIN/plugins")
   set_target_properties(_pyexatn PROPERTIES LINK_FLAGS "-shared")
endif()

install(TARGETS _pyexatn DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES exatn.py DESTINATION ${CMAKE_INSTALL_PREFIX})

if (EXATN_BUILD_TESTS)
   execute_process(COMMAND ${Python_EXECUTABLE} -c "import numpy" RESULT_VARIABLE NUMPY_EXISTS)
   if (NUMPY_EXISTS EQUAL "1")
	message(STATUS "Numpy not found. Skipping ExaTN Python Tester Build. It is recommended that you install numpy if using the ExaTN Python API.")
   else()
     message(STATUS "Found Numpy. Building ExaTN Python Tester.")
     add_subdirectory(tests)
   endif()
endif()