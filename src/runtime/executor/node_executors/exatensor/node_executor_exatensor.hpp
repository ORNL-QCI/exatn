/** ExaTN:: Tensor Runtime: Tensor graph node executor: Exatensor
REVISION: 2019/08/15

Copyright (C) 2018-2019 Dmitry Lyakh, Tiffany Mintz, Alex McCaskey
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle)

Rationale:

**/

#ifndef EXATN_RUNTIME_EXATENSOR_NODE_EXECUTOR_HPP_
#define EXATN_RUNTIME_EXATENSOR_NODE_EXECUTOR_HPP_

#include "tensor_node_executor.hpp"

namespace exatn {
namespace runtime {

class ExatensorNodeExecutor : public TensorNodeExecutor {

public:

  NodeExecHandleType execute(numerics::TensorOpCreate & op) override;
  NodeExecHandleType execute(numerics::TensorOpDestroy & op) override;
  NodeExecHandleType execute(numerics::TensorOpTransform & op) override;
  NodeExecHandleType execute(numerics::TensorOpAdd & op) override;
  NodeExecHandleType execute(numerics::TensorOpContract & op) override;

  bool sync(NodeExecHandleType op_handle, bool wait) override;

  const std::string name() const override {return "exatensor-node-executor";}
  const std::string description() const override {return "ExaTENSOR tensor graph node executor";}
  std::shared_ptr<TensorNodeExecutor> clone() override {return std::make_shared<ExatensorNodeExecutor>();}

protected:
 //`ExaTENSOR executor state
};

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_EXATENSOR_NODE_EXECUTOR_HPP_
