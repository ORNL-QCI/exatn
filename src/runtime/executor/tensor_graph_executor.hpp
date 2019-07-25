/** ExaTN:: Tensor Runtime: Tensor graph executor
REVISION: 2019/07/25

Copyright (C) 2018-2019 Dmitry Lyakh, Tiffany Mintz, Alex McCaskey
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle)

Rationale:

**/

#ifndef EXATN_RUNTIME_TENSOR_GRAPH_EXECUTOR_HPP_
#define EXATN_RUNTIME_TENSOR_GRAPH_EXECUTOR_HPP_

#include "Identifiable.hpp"

#include "tensor_graph.hpp"
#include "tensor_node_executor.hpp"
#include "tensor_operation.hpp"

#include <memory>

namespace exatn {
namespace runtime {

class TensorGraphExecutor : public Identifiable, public Cloneable<TensorGraphExecutor> {

public:

  /** Set the DAG node executor (tensor operation executor). **/
  bool setNodeExecutor(std::shared_ptr<TensorNodeExecutor> node_executor)
  {
    if(node_executor_) return false;
    node_executor_ = node_executor;
    return true;
  }

  /** Traverses the DAG and executes all its nodes. **/
  virtual void execute(TensorGraph & dag) = 0;

  virtual std::shared_ptr<TensorGraphExecutor> clone() = 0;

protected:
  std::shared_ptr<TensorNodeExecutor> node_executor_;
};

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_TENSOR_GRAPH_EXECUTOR_HPP_
