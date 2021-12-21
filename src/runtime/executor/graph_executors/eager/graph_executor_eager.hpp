/** ExaTN:: Tensor Runtime: Tensor graph executor: Eager
REVISION: 2021/12/21

Copyright (C) 2018-2021 Tiffany Mintz, Dmitry Lyakh, Alex McCaskey
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle)

Rationale:

**/

#ifndef EXATN_RUNTIME_EAGER_GRAPH_EXECUTOR_HPP_
#define EXATN_RUNTIME_EAGER_GRAPH_EXECUTOR_HPP_

#include "tensor_graph_executor.hpp"

namespace exatn {
namespace runtime {

class EagerGraphExecutor : public TensorGraphExecutor {

public:

  //EagerGraphExecutor(const EagerGraphExecutor &) = delete;
  //EagerGraphExecutor & operator=(const EagerGraphExecutor &) = delete;
  //EagerGraphExecutor(EagerGraphExecutor &&) = delete;
  //EagerGraphExecutor & operator=(EagerGraphExecutor &&) = delete;

  virtual ~EagerGraphExecutor() = default;

  /** Traverses the DAG and executes all its nodes. **/
  virtual void execute(TensorGraph & dag) override;

  /** Traverses the list of tensor networks and executes them as a whole. **/
  virtual void execute(TensorNetworkQueue & tensor_network_queue) override;

  /** Regulates the tensor prefetch depth (0 turns prefetch off). **/
  virtual void setPrefetchDepth(unsigned int depth) override {
    return;
  }

  const std::string name() const override {return "eager-dag-executor";}
  const std::string description() const override {return "Eager tensor graph executor";}
  std::shared_ptr<TensorGraphExecutor> clone() override {return std::make_shared<EagerGraphExecutor>();}
};

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_EAGER_GRAPH_EXECUTOR_HPP_
