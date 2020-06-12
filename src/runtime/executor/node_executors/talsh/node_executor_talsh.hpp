/** ExaTN:: Tensor Runtime: Tensor graph node executor: Talsh
REVISION: 2020/06/12

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
#include <atomic>

namespace exatn {
namespace runtime {

class TalshNodeExecutor : public TensorNodeExecutor {

public:

  static constexpr const std::size_t DEFAULT_MEM_BUFFER_SIZE = 1UL * 1024UL * 1024UL * 1024UL; //bytes

  TalshNodeExecutor() = default;
  TalshNodeExecutor(const TalshNodeExecutor &) = delete;
  TalshNodeExecutor & operator=(const TalshNodeExecutor &) = delete;
  TalshNodeExecutor(TalshNodeExecutor &&) noexcept = delete;
  TalshNodeExecutor & operator=(TalshNodeExecutor &&) noexcept = delete;
  virtual ~TalshNodeExecutor();

  void initialize(const ParamConf & parameters) override;

  std::size_t getMemoryBufferSize() const override;

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
  int execute(numerics::TensorOpBroadcast & op,
              TensorOpExecHandle * exec_handle) override;
  int execute(numerics::TensorOpAllreduce & op,
              TensorOpExecHandle * exec_handle) override;

  bool sync(TensorOpExecHandle op_handle,
            int * error_code,
            bool wait = true) override;

  bool sync(bool wait = true) override;

  bool discard(TensorOpExecHandle op_handle) override;

  bool prefetch(const numerics::TensorOperation & op) override;

  std::shared_ptr<talsh::Tensor> getLocalTensor(const numerics::Tensor & tensor,
                 const std::vector<std::pair<DimOffset,DimExtent>> & slice_spec) override;

  /** Finishes tensor operand prefetching for a given tensor operation. **/
  bool finishPrefetching(const numerics::TensorOperation & op);

  /** Caches TAL-SH tensor images moved to accelerators during tensor operation. **/
  void cacheMovedTensors(const talsh::TensorTask & talsh_task);

  /** Evicts some or all idle cached tensor images from an accelerator, moving them back to Host. **/
  void evictMovedTensors(int device_id = DEV_DEFAULT,     //in: flat device id (TAL-SH numeration), DEV_DEFAULT covers all accelerators
                         std::size_t required_space = 0); //in: required space to free in bytes (0 will evict all idle tensor images on the chosen device)

  const std::string name() const override {return "talsh-node-executor";}
  const std::string description() const override {return "TALSH tensor graph node executor";}
  std::shared_ptr<TensorNodeExecutor> clone() override {return std::make_shared<TalshNodeExecutor>();}

protected:
  /** Maps generic exatn::numerics::Tensor to its TAL-SH implementation talsh::Tensor **/
  std::unordered_map<numerics::TensorHashType,std::shared_ptr<talsh::Tensor>> tensors_;
  /** Active execution handles associated with tensor operations currently executed by TAL-SH **/
  std::unordered_map<TensorOpExecHandle,std::shared_ptr<talsh::TensorTask>> tasks_;
  /** Active tensor operand prefetching tasks **/
  std::unordered_map<numerics::TensorHashType,std::shared_ptr<talsh::TensorTask>> prefetches_;
  /** TAL-SH Host memory buffer size (bytes) **/
  std::atomic<std::size_t> talsh_host_mem_buffer_size_;
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
    case TensorElementType::REAL32: talsh_data_kind = talsh::REAL32; break;
    case TensorElementType::REAL64: talsh_data_kind = talsh::REAL64; break;
    case TensorElementType::COMPLEX32: talsh_data_kind = talsh::COMPLEX32; break;
    case TensorElementType::COMPLEX64: talsh_data_kind = talsh::COMPLEX64; break;
  }
  return talsh_data_kind;
}

/** TAL-SH tensor element kind --> ExaTN tensor element kind converter **/
inline TensorElementType get_exatn_tensor_element_kind(int element_type)
{
  switch(element_type){
    case talsh::REAL32: return TensorElementType::REAL32;
    case talsh::REAL64: return TensorElementType::REAL64;
    case talsh::COMPLEX32: return TensorElementType::COMPLEX32;
    case talsh::COMPLEX64: return TensorElementType::COMPLEX64;
  }
  return TensorElementType::VOID;
}

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_TALSH_NODE_EXECUTOR_HPP_
