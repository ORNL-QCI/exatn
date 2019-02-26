#include "ExampleTensorMethod.hpp"

#include "talshxx.hpp"

#include <iostream>

namespace exatn {

namespace test {

void ExampleTensorMethod::pack(BytePacket& packet) {
   appendToBytePacket(&packet, datum);
}

void ExampleTensorMethod::unpack(BytePacket &packet) {
    extractFromBytePacket(&packet, datum);
}

int ExampleTensorMethod::apply(const TensorDenseBlock& local_tensor) {

    // local_tensor.num_dims=1;
    // local_tensor.data_kind = 1;
    // local_tensor.body_ptr = (void*) data;
    auto nElements = 10;// getDenseTensorVolume(local_tensor);

    for (int i = 0; i < nElements; i++) {
        if (local_tensor.data_kind == ::talsh::TensorData<double>::kind) {
            ((double*)local_tensor.body_ptr)[i] = datum;
        } else {
            std::cerr << "can't handle any other data type.\n";
        }
    }

    return 0;

}

}
}