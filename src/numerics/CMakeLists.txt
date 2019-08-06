set(LIBRARY_NAME exatn-numerics)

add_library(${LIBRARY_NAME}
            SHARED
            tensor_symbol.cpp
            basis_vector.cpp
            space_basis.cpp
            spaces.cpp
            space_register.cpp
            tensor_shape.cpp
            tensor_signature.cpp
            tensor_leg.cpp
            tensor.cpp
            tensor_connected.cpp
            tensor_operation.cpp
            tensor_op_create.cpp
            tensor_op_destroy.cpp
            tensor_op_transform.cpp
            tensor_op_add.cpp
            tensor_op_contract.cpp
            tensor_op_factory.cpp
            network_builder_mps.cpp
            network_build_factory.cpp
            tensor_network.cpp)

target_include_directories(${LIBRARY_NAME} PUBLIC .)
target_link_libraries(${LIBRARY_NAME} PUBLIC ExaTensor::ExaTensor)

exatn_configure_plugin_rpath(${LIBRARY_NAME})

add_dependencies(${LIBRARY_NAME} exatensor-build)

if(EXATN_BUILD_TESTS)
  add_subdirectory(tests)
endif()

file (GLOB HEADERS *.hpp)

install(FILES ${HEADERS} DESTINATION include/exatn)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
