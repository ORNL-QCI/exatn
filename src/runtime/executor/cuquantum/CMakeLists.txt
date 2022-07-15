set(LIBRARY_NAME exatn_runtime_cuquantum)

file(GLOB SRC cuquantum_executor.cu)

set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -shared -std=c++14 -gencode arch=compute_${CUDA_ARCH_BIN},code=sm_${CUDA_ARCH_BIN} -rdc=true --compiler-options -fPIC")

cuda_add_library(${LIBRARY_NAME} SHARED ${SRC})

if(CUQUANTUM)
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES
                        CUDA_SEPARABLE_COMPILATION ON
                        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

set_target_properties(${LIBRARY_NAME} PROPERTIES DEFINE_SYMBOL "")

target_include_directories(${LIBRARY_NAME}
  PUBLIC .
  ${CMAKE_SOURCE_DIR}/src/exatn
  ${CMAKE_SOURCE_DIR}/src/utils
)

target_link_libraries(${LIBRARY_NAME} PUBLIC exatn-numerics)

if(CUQUANTUM)
  target_include_directories(${LIBRARY_NAME} PRIVATE ${CUQUANTUM_PATH}/include)
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${CUQUANTUM_PATH}/lib/libcutensornet.so)
endif()

if(CUTENSOR AND NOT CUTENSOR_PATH STREQUAL ".")
  target_include_directories(${LIBRARY_NAME} PRIVATE ${CUTENSOR_PATH}/include)
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${CUTENSOR_PATH}/lib/libcutensor.so ExaTensor::ExaTensor)
endif()

exatn_configure_plugin_rpath(${LIBRARY_NAME})

if(EXATN_BUILD_TESTS)
  #add_subdirectory(boost/tests)
endif()

file (GLOB HEADERS *.hpp)

install(FILES ${HEADERS} DESTINATION include/exatn)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
