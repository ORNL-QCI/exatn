/** ExaTN:: Tensor Runtime: Tensor graph node executor: Talsh
REVISION: 2021/07/23

Copyright (C) 2018-2021 Dmitry Lyakh, Tiffany Mintz, Alex McCaskey
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle)

Rationale:

**/

#ifndef EXATN_RUNTIME_TALSH_NODE_EXECUTOR_HPP_
#define EXATN_RUNTIME_TALSH_NODE_EXECUTOR_HPP_

#include "tensor_node_executor.hpp"

#include "talshxx.hpp"

#include <unordered_map>
#include <vector>
#include <list>
#include <memory>
#include <atomic>

namespace exatn {
namespace runtime {

class TalshNodeExecutor : public TensorNodeExecutor {

public:

  static constexpr const std::size_t DEFAULT_MEM_BUFFER_SIZE = 2UL * 1024UL * 1024UL * 1024UL; //bytes
  static constexpr const int ALLREDUCE_CHUNK_SIZE = 64 * 1024 * 1024; //elements

  TalshNodeExecutor(): max_tensor_rank_(-1), prefetch_enabled_(true), dry_run_(false) {}

  TalshNodeExecutor(const TalshNodeExecutor &) = delete;
  TalshNodeExecutor & operator=(const TalshNodeExecutor &) = delete;
  TalshNodeExecutor(TalshNodeExecutor &&) noexcept = delete;
  TalshNodeExecutor & operator=(TalshNodeExecutor &&) noexcept = delete;

  virtual ~TalshNodeExecutor();

  void initialize(const ParamConf & parameters) override;

  void activateDryRun(bool dry_run) override;

  void activateFastMath() override;

  std::size_t getMemoryBufferSize() const override;

  double getTotalFlopCount() const override;

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
  int execute(numerics::TensorOpFetch & op,
              TensorOpExecHandle * exec_handle) override;
  int execute(numerics::TensorOpUpload & op,
              TensorOpExecHandle * exec_handle) override;
  int execute(numerics::TensorOpBroadcast & op,
              TensorOpExecHandle * exec_handle) override;
  int execute(numerics::TensorOpAllreduce & op,
              TensorOpExecHandle * exec_handle) override;

  bool sync(TensorOpExecHandle op_handle,
            int * error_code,
            bool wait = true) override;

  bool sync() override;

  bool discard(TensorOpExecHandle op_handle) override;

  bool prefetch(const numerics::TensorOperation & op) override;

  /** Returns a locally stored slice copy of a tensor, or nullptr if no RAM. **/
  std::shared_ptr<talsh::Tensor> getLocalTensor(const numerics::Tensor & tensor,
                 const std::vector<std::pair<DimOffset,DimExtent>> & slice_spec) override;

  /** Finishes tensor operand prefetching for a given tensor operation. **/
  bool finishPrefetching(const numerics::TensorOperation & op); //in: tensor operation

  /** Caches TAL-SH tensor body images moved/copied to accelerators.  **/
  void cacheMovedTensors(talsh::TensorTask & talsh_task); //in: TAL-SH task associated with the tensor operation

  /** Evicts some or all idle cached TAL-SH tensor body images
      from accelerator(s), moving them back to Host. On return,
      returns whether at least one such tensor image has been found. **/
  bool evictMovedTensors(int device_id = DEV_DEFAULT,     //in: flat device id (TAL-SH numeration), DEV_DEFAULT covers all accelerators, DEV_HOST has no effect
                         std::size_t required_space = 0); //in: required space to free in bytes, 0 will evict all idle tensor images on the chosen device(s)

  const std::string name() const override {return "talsh-node-executor";}
  const std::string description() const override {return "TALSH tensor graph node executor";}
  std::shared_ptr<TensorNodeExecutor> clone() override {return std::make_shared<TalshNodeExecutor>();}

protected:

  /** Determines whether a given TAL-SH tensor is currently participating
      in an active tensor operation, tensor prefetch or tensor eviction. **/
  bool tensorIsCurrentlyInUse(const talsh::Tensor * talsh_tens) const;

  struct TensorImpl{
    //TAL-SH tensor with reduced shape (all extent-1 tensor dimensions removed):
    std::unique_ptr<talsh::Tensor> talsh_tensor;
    //The original full tensor signature (dimension base offsets):
    std::vector<std::size_t> full_base_offsets;
    //The reduced tensor signature (dimension base offsets):
    std::vector<std::size_t> reduced_base_offsets;
    //The original full tensor shape:
    talsh_tens_shape_t * stored_shape;
    //Flag which tensor shape is currently in use by the TAL-SH tensor:
    bool full_shape_is_on;
    //Lifecycle:
    TensorImpl(const std::vector<std::size_t> & full_offsets,    //full tensor signature
               const std::vector<DimExtent> & full_extents,      //full tensor shape
               const std::vector<std::size_t> & reduced_offsets, //reduced tensor signature
               const std::vector<int> & reduced_extents,         //reduced tensor shape
               int data_kind);                                   //TAL-SH tensor data kind
    TensorImpl(const TensorImpl &) = delete;
    TensorImpl & operator=(const TensorImpl &) = delete;
    TensorImpl(TensorImpl &&) noexcept;
    TensorImpl & operator=(TensorImpl &&) noexcept;
    ~TensorImpl();
    //Resets TAL-SH tensor shape between full and reduced, depending on the operation needs:
    void resetTensorShapeToFull();
    void resetTensorShapeToReduced();
  };

  struct CachedAttr{
    double last_used; //time stamp of last usage of the cached tensor image
  };

  /** Maps generic exatn::numerics::Tensor to its TAL-SH implementation **/
  std::unordered_map<numerics::TensorHashType,TensorImpl> tensors_;
  /** Active execution handles associated with tensor operations currently executed by TAL-SH **/
  std::unordered_map<TensorOpExecHandle,std::shared_ptr<talsh::TensorTask>> tasks_;
  /** Active tensor operand prefetching to accelerators tasks **/
  std::unordered_map<numerics::TensorHashType,std::shared_ptr<talsh::TensorTask>> prefetches_;
  /** Active tensor image eviction from accelerators tasks **/
  std::unordered_map<talsh::Tensor*,std::shared_ptr<talsh::TensorTask>> evictions_;
  /** Register (cache) of tensors with body images moved/copied to accelerators **/
  std::unordered_map<talsh::Tensor*,CachedAttr> accel_cache_[DEV_MAX]; //cache for each device
  /** Active MPI requests for non-blocking two-sided messages **/
  std::unordered_map<TensorOpExecHandle,std::list<void*>> mpi_requests_; //owning pointers
  /** Max encountered actual tensor rank **/
  int max_tensor_rank_;
  /** Prefetching enabled flag **/
  bool prefetch_enabled_;
  /** Dry run (no actual computations) **/
  std::atomic<bool> dry_run_;
  /** TAL-SH Host memory buffer size (bytes) **/
  static std::atomic<std::size_t> talsh_host_mem_buffer_size_;
  /** TAL-SH submitted Flop count **/
  static std::atomic<double> talsh_submitted_flops_;
  /** TAL-SH initialization status **/
  static std::atomic<bool> talsh_initialized_;
  /** Number of instances of TAL-SH node executors **/
  static std::atomic<int> talsh_node_exec_count_;
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
