#ifndef EXATN_RUNTIME_GRAPH_EXECUTOR_HPP_
#define EXATN_RUNTIME_GRAPH_EXECUTOR_HPP_

#include "tensor_graph.hpp"
#include "tensor_operation.hpp"

#include <memory>

namespace exatn {
namespace runtime {

class GraphExecutor {

public:

  void execute(TensorGraph & dag);

protected:

  virtual void exec_impl(TensorOperation & op) = 0;

  int nextExecutableNodeId(TensorGraph & dag);
};

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_GRAPH_EXECUTOR_HPP_
