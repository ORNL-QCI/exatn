#include "ExampleTensorMethod.hpp"


namespace exatn {

namespace test {

void ExampleTensorMethod::pack(BytePacket& packet) {
   appendToBytePacket(&packet, data);
}

void ExampleTensorMethod::unpack(BytePacket &packet) {
    extractFromBytePacket(&packet, data);
}

int ExampleTensorMethod::apply(const TensorDenseBlock& local_tensor) {

    // local_tensor.num_dims=1;
    // local_tensor.data_kind = 1;
    // local_tensor.body_ptr = (void*) data;

    return 0;

}

}
}