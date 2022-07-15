set(LIBRARY_NAME exatn-numerics)

set(INTERFACE_POSITION_INDEPENDENT_CODE ON)

file(GLOB SRC *.cpp)

if(CUQUANTUM)
  enable_language(CUDA)
  set(CUDA_FOUND TRUE)
  set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++14 -gencode arch=compute_${CUDA_ARCH_BIN},code=sm_${CUDA_ARCH_BIN} -rdc=true --compiler-options -fPIC ${CMAKE_CXX_FLAGS}")
  set_source_files_properties(contraction_seq_optimizer_cutnn.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_library(${LIBRARY_NAME} SHARED ${SRC})

if(CUQUANTUM)
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES
                        CUDA_SEPARABLE_COMPILATION ON
                        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

#set_target_properties(${LIBRARY_NAME} PROPERTIES DEFINE_SYMBOL "")

target_include_directories(${LIBRARY_NAME}
                           PUBLIC .
                           PRIVATE
                           ${CMAKE_SOURCE_DIR}/tpls/metis/include
                           ${CMAKE_SOURCE_DIR}/src/runtime
                           ${CMAKE_SOURCE_DIR}/src/runtime/executor
                           ${CMAKE_SOURCE_DIR}/src/exatn)

target_link_libraries(${LIBRARY_NAME} PUBLIC ExaTensor::ExaTensor exatn-utils metis GKlib)

if(CUQUANTUM)
  target_include_directories(${LIBRARY_NAME} PRIVATE ${CUQUANTUM_PATH}/include)
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${CUQUANTUM_PATH}/lib/libcutensornet.so)
endif()

if(CUTENSOR AND NOT CUTENSOR_PATH STREQUAL ".")
  target_include_directories(${LIBRARY_NAME} PRIVATE ${CUTENSOR_PATH}/include)
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${CUTENSOR_PATH}/lib/libcutensor.so)
endif()

exatn_configure_plugin_rpath(${LIBRARY_NAME})

add_dependencies(${LIBRARY_NAME} exatensor-build)

if(EXATN_BUILD_TESTS)
  add_subdirectory(tests)
endif()

file (GLOB HEADERS *.hpp)

install(FILES ${HEADERS} DESTINATION include/exatn)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
