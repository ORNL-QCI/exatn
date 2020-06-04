/** ExaTN:: Tensor Runtime: Tensor graph executor: Lazy
REVISION: 2020/06/04

Copyright (C) 2018-2020 Dmitry Lyakh, Tiffany Mintz, Alex McCaskey
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle)

Rationale:

**/

#ifndef EXATN_RUNTIME_LAZY_GRAPH_EXECUTOR_HPP_
#define EXATN_RUNTIME_LAZY_GRAPH_EXECUTOR_HPP_

#include "tensor_graph_executor.hpp"

namespace exatn {
namespace runtime {

class LazyGraphExecutor : public TensorGraphExecutor {

public:

  virtual ~LazyGraphExecutor() = default;

  /** Traverses the DAG and executes all its nodes. **/
  void execute(TensorGraph & dag) override;

  const std::string name() const override {return "lazy-dag-executor";}
  const std::string description() const override {return "Lazy tensor graph executor";}
  std::shared_ptr<TensorGraphExecutor> clone() override {return std::make_shared<LazyGraphExecutor>();}
};

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_LAZY_GRAPH_EXECUTOR_HPP_
