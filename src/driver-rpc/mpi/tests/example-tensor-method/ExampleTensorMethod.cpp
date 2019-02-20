#include "ExampleTensorMethod.hpp"


namespace exatn {

namespace test {

void ExampleTensorMethod::pack(BytePacket& packet) {
    packet.base_addr = &data;
    packet.size_bytes = sizeof (data); //size;
}

void ExampleTensorMethod::unpack(const BytePacket &packet) {
    data = (int&) packet.base_addr;
    size = packet.size_bytes;
}

int ExampleTensorMethod::apply(const TensorDenseBlock& local_tensor) {

    // local_tensor.num_dims=1;
    // local_tensor.data_kind = 1;
    // local_tensor.body_ptr = (void*) data;

    return 0;

}

}
}