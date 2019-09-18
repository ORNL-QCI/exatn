#include "tensor_method.hpp"
#include "Identifiable.hpp"

namespace exatn {
namespace test {

class ExampleTensorMethod : public talsh::TensorFunctor<Identifiable> {

protected:

  double datum = 0.42;

public:

  void pack(BytePacket & packet) override;
  void unpack(BytePacket & packet) override;
  int apply(talsh::Tensor & local_tensor) override;

  const std::string name() const override {return "HamiltonianTest";}
  const std::string description() const override {return "";}
};

}
}
