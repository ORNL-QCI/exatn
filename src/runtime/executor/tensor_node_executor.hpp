/** ExaTN:: Tensor Runtime: Tensor graph node executor
REVISION: 2019/08/26

Copyright (C) 2018-2019 Dmitry Lyakh, Tiffany Mintz, Alex McCaskey
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle)

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

#include <memory>

namespace exatn {
namespace runtime {

// Tensor operation execution handle:
using TensorOpExecHandle = std::size_t;


class TensorNodeExecutor : public Identifiable, public Cloneable<TensorNodeExecutor> {

public:

  /** Executes the tensor operation found in a DAG node. **/
  virtual TensorOpExecHandle execute(numerics::TensorOpCreate & op) = 0;
  virtual TensorOpExecHandle execute(numerics::TensorOpDestroy & op) = 0;
  virtual TensorOpExecHandle execute(numerics::TensorOpTransform & op) = 0;
  virtual TensorOpExecHandle execute(numerics::TensorOpAdd & op) = 0;
  virtual TensorOpExecHandle execute(numerics::TensorOpContract & op) = 0;

  /** Synchronizes the execution of a previously submitted tensor operation. **/
  virtual bool sync(TensorOpExecHandle op_handle,
                    int * error_code,
                    bool wait = false) = 0;

  virtual std::shared_ptr<TensorNodeExecutor> clone() = 0;
};

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_TENSOR_NODE_EXECUTOR_HPP_
