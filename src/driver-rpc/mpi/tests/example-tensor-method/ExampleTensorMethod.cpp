#include "ExampleTensorMethod.hpp"
#include "talshxx.hpp"

#include <iostream>

namespace exatn {
namespace test {

void ExampleTensorMethod::pack(BytePacket & packet) {
  appendToBytePacket(&packet,datum);
}

void ExampleTensorMethod::unpack(BytePacket & packet) {
  extractFromBytePacket(&packet,datum);
}

int ExampleTensorMethod::apply(talsh::Tensor & local_tensor) {

  auto nElements = local_tensor.getVolume();

  double * elements;
  auto is_double = local_tensor.getDataAccessHost(&elements);
  assert(is_double);

  for(decltype(nElements) i = 0; i < nElements; i++){
    elements[i] = 0.0;
  }

  return 0;
}

} //namespace test
} //namespace exatn
