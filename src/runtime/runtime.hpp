#ifndef EXATN_RUNTIME_HPP_
#define EXATN_RUNTIME_HPP_

#include <iostream>
#include <memory>

namespace exatn {
namespace runtime {
class TensorRuntime {

public:

  void submit(TensorOp& op) {
    // add on to the graph
  }

  void sync(const exatn::numerics::Tensor& tensor) {
      // sync on a particular tensor, everything related to tensor
      // must complete
  }

  TensorDenseBlock getTensorData(const exatn::numerics::Tensor& tensor) {
    // get tensor data after sync
  }
};

}
}
#endif
