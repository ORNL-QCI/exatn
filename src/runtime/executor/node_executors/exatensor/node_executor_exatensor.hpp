/** ExaTN:: Tensor Runtime: Tensor graph node executor: Exatensor
REVISION: 2020/04/07

Copyright (C) 2018-2020 Dmitry Lyakh, Tiffany Mintz, Alex McCaskey
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle)

Rationale:

**/

#ifndef EXATN_RUNTIME_EXATENSOR_NODE_EXECUTOR_HPP_
#define EXATN_RUNTIME_EXATENSOR_NODE_EXECUTOR_HPP_

#include "tensor_node_executor.hpp"

#include "talshxx.hpp"

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

  const std::string name() const override {return "exatensor-node-executor";}
  const std::string description() const override {return "ExaTENSOR tensor graph node executor";}
  std::shared_ptr<TensorNodeExecutor> clone() override {return std::make_shared<ExatensorNodeExecutor>();}

protected:
 //`ExaTENSOR executor state
};

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_EXATENSOR_NODE_EXECUTOR_HPP_
