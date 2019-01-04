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
target_link_libraries(${LIBRARY_NAME} exatn-utility exatensor talsh)

if(EXATN_BUILD_TESTS)
  add_subdirectory(tests)
endif()
