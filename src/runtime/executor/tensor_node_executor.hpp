/** ExaTN:: Tensor Runtime: Tensor graph node executor
REVISION: 2020/02/28

Copyright (C) 2018-2020 Dmitry Lyakh, Tiffany Mintz, Alex McCaskey
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle)

Rationale:
 (a) Tensor node executor provides actual implementation of registered
     tensor operations. Actual tensor operations are submitted for
     generally asynchronous execution via the .execute method overloads,
     which return an asynchronous execution handle assoicated with the
     submitted tensor operation. After submission, the completion status
     of the the tensor operation can be checked or enforced via the .sync
     method by providing the asynchronous execution handle previously
     returned by the .submit method.
**/

#ifndef EXATN_RUNTIME_TENSOR_NODE_EXECUTOR_HPP_
#define EXATN_RUNTIME_TENSOR_NODE_EXECUTOR_HPP_

#include "Identifiable.hpp"

#include "tensor_op_factory.hpp"
#include "tensor.hpp"
#include "space_register.hpp"

#include <vector>
#include <memory>

namespace talsh{
class Tensor;
}

namespace exatn {
namespace runtime {

class TensorNodeExecutor : public Identifiable, public Cloneable<TensorNodeExecutor> {

public:

  virtual ~TensorNodeExecutor() = default;

  /** Explicitly initializes the underlying numerical service, if needed **/
  virtual void initialize() = 0;

  /** Executes the tensor operation found in a DAG node asynchronously,
      returning the execution handle in exec_handle that can later be
      used for testing for completion of the operation execution.
      Returns an integer error code (0:Success). **/
  virtual int execute(numerics::TensorOpCreate & op,
                      TensorOpExecHandle * exec_handle) = 0;
  virtual int execute(numerics::TensorOpDestroy & op,
                      TensorOpExecHandle * exec_handle) = 0;
  virtual int execute(numerics::TensorOpTransform & op,
                      TensorOpExecHandle * exec_handle) = 0;
  virtual int execute(numerics::TensorOpSlice & op,
                      TensorOpExecHandle * exec_handle) = 0;
  virtual int execute(numerics::TensorOpInsert & op,
                      TensorOpExecHandle * exec_handle) = 0;
  virtual int execute(numerics::TensorOpAdd & op,
                      TensorOpExecHandle * exec_handle) = 0;
  virtual int execute(numerics::TensorOpContract & op,
                      TensorOpExecHandle * exec_handle) = 0;

  /** Synchronizes the execution of a previously submitted tensor operation. **/
  virtual bool sync(TensorOpExecHandle op_handle,
                    int * error_code,
                    bool wait = false) = 0;

  /** Returns a local copy of a given tensor slice. **/
  virtual std::shared_ptr<talsh::Tensor> getLocalTensor(const numerics::Tensor & tensor,
                         const std::vector<std::pair<DimOffset,DimExtent>> & slice_spec) = 0;

  virtual std::shared_ptr<TensorNodeExecutor> clone() = 0;
};

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_TENSOR_NODE_EXECUTOR_HPP_
