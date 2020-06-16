/** ExaTN:: Tensor Runtime: Tensor graph executor: Lazy
REVISION: 2020/06/16

Copyright (C) 2018-2020 Dmitry Lyakh, Alex McCaskey
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

  static constexpr const unsigned int DEFAULT_PIPELINE_DEPTH = 16;
  static constexpr const unsigned int DEFAULT_PREFETCH_DEPTH = 4;

  LazyGraphExecutor(): pipeline_depth_(DEFAULT_PIPELINE_DEPTH),
                       prefetch_depth_(DEFAULT_PREFETCH_DEPTH) {}

  virtual ~LazyGraphExecutor() = default;

  /** Traverses the DAG and executes all its nodes. **/
  void execute(TensorGraph & dag) override;

  /** Returns the current pipeline depth. **/
  inline unsigned int getPipelineDepth() const {return pipeline_depth_;}

  /** Returns the current prefetch depth. **/
  inline unsigned int getPrefetchDepth() const {return prefetch_depth_;}

  const std::string name() const override {return "lazy-dag-executor";}
  const std::string description() const override {return "Lazy tensor graph executor";}
  std::shared_ptr<TensorGraphExecutor> clone() override {return std::make_shared<LazyGraphExecutor>();}

protected:

 unsigned int pipeline_depth_; //max number of active tensor operations in flight
 unsigned int prefetch_depth_; //max number of tensor operations with active prefetch
};

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_LAZY_GRAPH_EXECUTOR_HPP_
