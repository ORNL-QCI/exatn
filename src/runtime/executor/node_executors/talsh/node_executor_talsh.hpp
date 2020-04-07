/** ExaTN:: Tensor Runtime: Tensor graph node executor: Talsh
REVISION: 2020/04/07

Copyright (C) 2018-2020 Dmitry Lyakh, Tiffany Mintz, Alex McCaskey
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle)

Rationale:

**/

#ifndef EXATN_RUNTIME_TALSH_NODE_EXECUTOR_HPP_
#define EXATN_RUNTIME_TALSH_NODE_EXECUTOR_HPP_

#include "tensor_node_executor.hpp"

#include "talshxx.hpp"

#include <unordered_map>
#include <vector>
#include <memory>

namespace exatn {
namespace runtime {

class TalshNodeExecutor : public TensorNodeExecutor {

public:

  TalshNodeExecutor() = default;
  TalshNodeExecutor(const TalshNodeExecutor &) = delete;
  TalshNodeExecutor & operator=(const TalshNodeExecutor &) = delete;
  TalshNodeExecutor(TalshNodeExecutor &&) noexcept = delete;
  TalshNodeExecutor & operator=(TalshNodeExecutor &&) noexcept = delete;
  virtual ~TalshNodeExecutor();

  void initialize() override;

  int execute(numerics::TensorOpCreate & op,
              TensorOpExecHandle * exec_handle) override;
  int execute(numerics::TensorOpDestroy & op,
              TensorOpExecHandle * exec_handle) override;
  int execute(numerics::TensorOpTransform & op,
              TensorOpExecHandle * exec_handle) override;
  int execute(numerics::TensorOpSlice & op,
              TensorOpExecHandle * exec_handle) override;
  int execute(numerics::TensorOpInsert & op,
              TensorOpExecHandle * exec_handle) override;
  int execute(numerics::TensorOpAdd & op,
              TensorOpExecHandle * exec_handle) override;
  int execute(numerics::TensorOpContract & op,
              TensorOpExecHandle * exec_handle) override;
  int execute(numerics::TensorOpDecomposeSVD3 & op,
              TensorOpExecHandle * exec_handle) override;
  int execute(numerics::TensorOpDecomposeSVD2 & op,
              TensorOpExecHandle * exec_handle) override;
  int execute(numerics::TensorOpOrthogonalizeSVD & op,
              TensorOpExecHandle * exec_handle) override;
  int execute(numerics::TensorOpOrthogonalizeMGS & op,
              TensorOpExecHandle * exec_handle) override;

  bool sync(TensorOpExecHandle op_handle,
            int * error_code,
            bool wait = false) override;

  std::shared_ptr<talsh::Tensor> getLocalTensor(const numerics::Tensor & tensor,
                 const std::vector<std::pair<DimOffset,DimExtent>> & slice_spec) override;

  const std::string name() const override {return "talsh-node-executor";}
  const std::string description() const override {return "TALSH tensor graph node executor";}
  std::shared_ptr<TensorNodeExecutor> clone() override {return std::make_shared<TalshNodeExecutor>();}

protected:
  /** Maps generic exatn::numerics::Tensor to its TAL-SH implementation talsh::Tensor **/
  std::unordered_map<numerics::TensorHashType,std::shared_ptr<talsh::Tensor>> tensors_;
  /** Active execution handles associated with tensor operations currently executed by TAL-SH **/
  std::unordered_map<TensorOpExecHandle,std::shared_ptr<talsh::TensorTask>> tasks_;
  /** TAL-SH initialization status **/
  static bool talsh_initialized_;
  /** Number of instances of TAL-SH node executors **/
  static int talsh_node_exec_count_;
};


/** ExaTN tensor element kind --> TAL-SH tensor element kind converter **/
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

/** TAL-SH tensor element kind --> ExaTN tensor element kind converter **/
inline TensorElementType get_exatn_tensor_element_kind(int element_type)
{
  switch(element_type){
    case R4: return TensorElementType::REAL32;
    case R8: return TensorElementType::REAL64;
    case C4: return TensorElementType::COMPLEX32;
    case C8: return TensorElementType::COMPLEX64;
  }
  return TensorElementType::VOID;
}

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_TALSH_NODE_EXECUTOR_HPP_
