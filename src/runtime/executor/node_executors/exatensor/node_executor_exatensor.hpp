/** ExaTN:: Tensor Runtime: Tensor graph node executor: Exatensor
REVISION: 2021/01/28

Copyright (C) 2018-2021 Dmitry Lyakh, Tiffany Mintz, Alex McCaskey
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle)

Rationale:

**/

#ifndef EXATN_RUNTIME_EXATENSOR_NODE_EXECUTOR_HPP_
#define EXATN_RUNTIME_EXATENSOR_NODE_EXECUTOR_HPP_

#include "tensor_node_executor.hpp"

#include "talshxx.hpp"

#include <atomic>

namespace exatn {
namespace runtime {

class ExatensorNodeExecutor : public TensorNodeExecutor {

public:

  ExatensorNodeExecutor() = default;
  ExatensorNodeExecutor(const ExatensorNodeExecutor &) = delete;
  ExatensorNodeExecutor & operator=(const ExatensorNodeExecutor &) = delete;
  ExatensorNodeExecutor(ExatensorNodeExecutor &&) noexcept = delete;
  ExatensorNodeExecutor & operator=(ExatensorNodeExecutor &&) noexcept = delete;
  virtual ~ExatensorNodeExecutor() = default;

  void initialize(const ParamConf & parameters) override;

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

  std::shared_ptr<talsh::Tensor> getLocalTensor(const numerics::Tensor & tensor,
                 const std::vector<std::pair<DimOffset,DimExtent>> & slice_spec) override;

  const std::string name() const override {return "exatensor-node-executor";}
  const std::string description() const override {return "ExaTENSOR tensor graph node executor";}
  std::shared_ptr<TensorNodeExecutor> clone() override {return std::make_shared<ExatensorNodeExecutor>();}

protected:
 //`ExaTENSOR executor state
 /** Size of the distributed Host memory buffer provided by ExaTENSOR in bytes **/
 std::atomic<std::size_t> exatensor_host_mem_buffer_size_;
};

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_EXATENSOR_NODE_EXECUTOR_HPP_
