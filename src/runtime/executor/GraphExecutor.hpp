#ifndef EXATN_RUNTIME_GRAPH_EXECUTOR_HPP_
#define EXATN_RUNTIME_GRAPH_EXECUTOR_HPP_

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

   virtual void exec_impl(numerics::TensorOperation & op) = 0;

   int nextExecutableNodeId(TensorGraph & dag);

   std::mutex mtx;

};

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_GRAPH_EXECUTOR_HPP_
