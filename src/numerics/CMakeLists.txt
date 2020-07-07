set(LIBRARY_NAME exatn-numerics)

add_library(${LIBRARY_NAME}
            SHARED
            tensor_symbol.cpp
            metis_graph.cpp
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
            tensor_op_slice.cpp
            tensor_op_insert.cpp
            tensor_op_add.cpp
            tensor_op_contract.cpp
            tensor_op_decompose_svd3.cpp
            tensor_op_decompose_svd2.cpp
            tensor_op_orthogonalize_svd.cpp
            tensor_op_orthogonalize_mgs.cpp
            tensor_op_broadcast.cpp
            tensor_op_allreduce.cpp
            tensor_op_factory.cpp
            network_builder_mps.cpp
            network_builder_tree.cpp
            network_build_factory.cpp
            contraction_seq_optimizer.cpp
            contraction_seq_optimizer_dummy.cpp
            contraction_seq_optimizer_heuro.cpp
            contraction_seq_optimizer_greed.cpp
            contraction_seq_optimizer_metis.cpp
            contraction_seq_optimizer_factory.cpp
            tensor_network.cpp
            tensor_operator.cpp
            tensor_expansion.cpp
            functor_init_val.cpp
            functor_init_rnd.cpp
            functor_init_dat.cpp
            functor_scale.cpp
            functor_norm1.cpp
            functor_norm2.cpp
            functor_diag_rank.cpp)

target_include_directories(${LIBRARY_NAME}
                    PUBLIC .
                    PRIVATE
                           ${CMAKE_SOURCE_DIR}/tpls/metis/include
                           ${CMAKE_SOURCE_DIR}/src/runtime
                           ${CMAKE_SOURCE_DIR}/src/runtime/executor
                           ${CMAKE_SOURCE_DIR}/src/exatn)
target_link_libraries(${LIBRARY_NAME} PUBLIC ExaTensor::ExaTensor metis)

exatn_configure_plugin_rpath(${LIBRARY_NAME})

add_dependencies(${LIBRARY_NAME} exatensor-build)

if(EXATN_BUILD_TESTS)
  add_subdirectory(tests)
endif()

file (GLOB HEADERS *.hpp)

install(FILES ${HEADERS} DESTINATION include/exatn)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
