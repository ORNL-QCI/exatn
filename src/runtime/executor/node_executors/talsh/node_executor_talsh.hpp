/** ExaTN:: Tensor Runtime: Tensor graph node executor: Talsh
REVISION: 2019/08/15

Copyright (C) 2018-2019 Dmitry Lyakh, Tiffany Mintz, Alex McCaskey
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle)

Rationale:

**/

#ifndef EXATN_RUNTIME_TALSH_NODE_EXECUTOR_HPP_
#define EXATN_RUNTIME_TALSH_NODE_EXECUTOR_HPP_

#include "tensor_node_executor.hpp"

#include "talshxx.hpp"

#include <unordered_map>
#include <memory>

namespace exatn {
namespace runtime {

class TalshNodeExecutor : public TensorNodeExecutor {

public:

  NodeExecHandleType execute(numerics::TensorOpCreate & op) override;
  NodeExecHandleType execute(numerics::TensorOpDestroy & op) override;
  NodeExecHandleType execute(numerics::TensorOpTransform & op) override;
  NodeExecHandleType execute(numerics::TensorOpAdd & op) override;
  NodeExecHandleType execute(numerics::TensorOpContract & op) override;

  bool sync(NodeExecHandleType op_handle, bool wait) override;

  const std::string name() const override {return "talsh-node-executor";}
  const std::string description() const override {return "TALSH tensor graph node executor";}
  std::shared_ptr<TensorNodeExecutor> clone() override {return std::make_shared<TalshNodeExecutor>();}

protected:
  /** Maps generic exatn::numerics::Tensor to its TAL-SH implementation talsh::Tensor **/
  std::unordered_map<TensorHashType,talsh::Tensor> tensors_;
  /** Active handles associated with tensor operations currently executed by TAL-SH **/
  std::unordered_map<NodeExecHandleType,talsh::TensorTask> tasks_;
};

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_TALSH_NODE_EXECUTOR_HPP_
