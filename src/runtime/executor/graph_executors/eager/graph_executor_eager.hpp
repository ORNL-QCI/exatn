/** ExaTN:: Tensor Runtime: Tensor graph executor: Eager
REVISION: 2019/07/24

Copyright (C) 2018-2019 Tiffany Mintz, Dmitry Lyakh, Alex McCaskey
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle)

Rationale:

**/

#ifndef EXATN_RUNTIME_EAGER_GRAPH_EXECUTOR_HPP_
#define EXATN_RUNTIME_EAGER_GRAPH_EXECUTOR_HPP_

#include "tensor_graph_executor.hpp"

namespace exatn {
namespace runtime {

class EagerGraphExecutor : public TensorGraphExecutor {

public:

  /** Traverses the DAG and executes all its nodes. **/
  void execute(TensorGraph & dag) override;

  const std::string name() const override {return "eager-dag-executor";}
  const std::string description() const override {return "Eager tensor graph executor";}
  std::shared_ptr<TensorGraphExecutor> clone() override {return std::make_shared<EagerGraphExecutor>();}

protected:

  int nextExecutableNodeId(TensorGraph & dag);

};

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_EAGER_GRAPH_EXECUTOR_HPP_
