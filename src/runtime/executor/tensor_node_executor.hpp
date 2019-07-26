/** ExaTN:: Tensor Runtime: Tensor graph node executor
REVISION: 2019/07/26

Copyright (C) 2018-2019 Dmitry Lyakh, Tiffany Mintz, Alex McCaskey
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle)

Rationale:

**/

#ifndef EXATN_RUNTIME_TENSOR_NODE_EXECUTOR_HPP_
#define EXATN_RUNTIME_TENSOR_NODE_EXECUTOR_HPP_

#include "Identifiable.hpp"

#include "tensor_operation.hpp"

#include <memory>

namespace exatn {
namespace runtime {

// Tensor implementation:
using numerics::TensorHashType; //each numerics::Tensor has its unique integer hash
using numerics::Tensor;
using numerics::TensorOperation;

// DAG node execution handle (tensor operation execution handle):
using NodeExecHandleType = std::size_t;


class TensorNodeExecutor : public Identifiable, public Cloneable<TensorNodeExecutor> {

public:

  /** Executes the tensor operation found in a DAG node. **/
  virtual NodeExecHandleType execute(TensorOperation & op) = 0;

  /** Synchronizes the execution of a tensor operation. **/
  virtual bool sync(NodeExecHandleType op_handle, bool wait = false) = 0;

  virtual std::shared_ptr<TensorNodeExecutor> clone() = 0;

};

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_TENSOR_NODE_EXECUTOR_HPP_
