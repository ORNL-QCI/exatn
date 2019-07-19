#ifndef EXATN_RUNTIME_DAGOPT_HPP_
#define EXATN_RUNTIME_DAGOPT_HPP_

#include "TensorGraph.hpp"
#include "tensor_operation.hpp"

#include <iostream>
#include <memory>
#include <mutex>

namespace exatn {
namespace runtime {

// temp
using TensorOp = int;

class GraphExecutor {

public:

   void execute(TensorGraph& dag);

protected:

   virtual void exec_impl(numerics::TensorOperation& op) = 0;

   TensorOpNode nextExecutableNode(TensorGraph& dag);

   std::mutex mtx;

};
}
} // namespace exatn
#endif
