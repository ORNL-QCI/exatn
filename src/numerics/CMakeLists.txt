set(LIBRARY_NAME exatn-numerics)

add_library(${LIBRARY_NAME}
            SHARED
            tensor_signature.cpp
            tensor_shape.cpp
            tensor_leg.cpp
            tensor.cpp
            tensor_exa.cpp
            tensor_factory.cpp)

target_include_directories(${LIBRARY_NAME} PUBLIC .)

exatn_enable_rpath(${LIBRARY_NAME})

add_dependencies(${LIBRARY_NAME} exatensor-build)

if(EXATN_BUILD_TESTS)
  add_subdirectory(tests)
endif()

file (GLOB HEADERS *.hpp)

install(FILES ${HEADERS} DESTINATION include/exatn)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
