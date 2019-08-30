/** ExaTN:: Tensor Runtime: Tensor graph node executor: Talsh
REVISION: 2019/08/30

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

  int execute(numerics::TensorOpCreate & op,
              TensorOpExecHandle * exec_handle) override;
  int execute(numerics::TensorOpDestroy & op,
              TensorOpExecHandle * exec_handle) override;
  int execute(numerics::TensorOpTransform & op,
              TensorOpExecHandle * exec_handle) override;
  int execute(numerics::TensorOpAdd & op,
              TensorOpExecHandle * exec_handle) override;
  int execute(numerics::TensorOpContract & op,
              TensorOpExecHandle * exec_handle) override;

  bool sync(TensorOpExecHandle op_handle,
            int * error_code,
            bool wait = false) override;

  const std::string name() const override {return "talsh-node-executor";}
  const std::string description() const override {return "TALSH tensor graph node executor";}
  std::shared_ptr<TensorNodeExecutor> clone() override {return std::make_shared<TalshNodeExecutor>();}

  friend void check_initialize_talsh();

protected:
  /** Maps generic exatn::numerics::Tensor to its TAL-SH implementation talsh::Tensor **/
  std::unordered_map<numerics::TensorHashType,std::shared_ptr<talsh::Tensor>> tensors_;
  /** Active execution handles associated with tensor operations currently executed by TAL-SH **/
  std::unordered_map<TensorOpExecHandle,std::shared_ptr<talsh::TensorTask>> tasks_;
  /** TAL-SH initialization status **/
  static bool talsh_initialized_;
};


inline int get_talsh_tensor_element_kind(TensorElementType element_type)
{
 int talsh_data_kind = NO_TYPE;
 switch(element_type){
  case TensorElementType::REAL32: talsh_data_kind = R4; break;
  case TensorElementType::REAL64: talsh_data_kind = R8; break;
  case TensorElementType::COMPLEX32: talsh_data_kind = C4; break;
  case TensorElementType::COMPLEX64: talsh_data_kind = C8; break;
 }
 return talsh_data_kind;
}

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_TALSH_NODE_EXECUTOR_HPP_
