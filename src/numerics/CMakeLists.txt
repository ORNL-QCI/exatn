set(LIBRARY_NAME exatn-numerics)
add_library(${LIBRARY_NAME}
            SHARED
            tensor_factory.cpp
            tensor_leg.cpp
            tensor_shape.cpp
            tensor_signature.cpp
            tensor.cpp)
target_include_directories(${LIBRARY_NAME} PUBLIC .)
target_link_libraries(${LIBRARY_NAME} exatn-utility)

if(EXATN_BUILD_TESTS)
  add_subdirectory(tests)
endif()
