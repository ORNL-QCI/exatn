set(LIBRARY_NAME exatn-numerics)

add_library(${LIBRARY_NAME}
            SHARED
            basis_vector.cpp
            space_basis.cpp
            spaces.cpp
            tensor_signature.cpp
            tensor_shape.cpp
            tensor_leg.cpp
            tensor.cpp
            tensor_connected.cpp
            space_register.cpp
            tensor_operation.cpp
            tensor_op_create.cpp
            tensor_op_destroy.cpp
            tensor_op_transform.cpp
            tensor_op_add.cpp
            tensor_op_contract.cpp
            num_server.cpp)

target_include_directories(${LIBRARY_NAME}
                           PUBLIC .
                           ${CMAKE_SOURCE_DIR}/tpls/ExaTensor/include
                           ${CMAKE_SOURCE_DIR}/src/exatn
                           PRIVATE
                           ${CUDA_INCLUDE_DIRS})

exatn_enable_rpath(${LIBRARY_NAME})

add_dependencies(${LIBRARY_NAME} exatensor-build)

if(EXATN_BUILD_TESTS)
  add_subdirectory(tests)
endif()

file (GLOB HEADERS *.hpp)

install(FILES ${HEADERS} DESTINATION include/exatn)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
