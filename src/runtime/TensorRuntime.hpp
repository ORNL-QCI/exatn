/** ExaTN:: Tensor Runtime: Execution layer for tensor operations
REVISION: 2019/07/18

Copyright (C) 2018-2019 Tiffany Mintz, Dmitry Lyakh, Alex McCaskey
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle)

Rationale:
 (a) The execution space consists of one or more DAGs in which
     nodes represent tensor operations and directed edges represent
     dependencies between the corresponding nodes (tensor operations).
     Each DAG is associated with a uniquely named TAProL scope.
 (b) The DAG lifecycle:
     openScope(name): Opens a new TAProL scope and creates its associated empty DAG.
                      The .submit method can then be used to append new tensor
                      operations into the current DAG. The actual execution
                      of the submitted tensor operations may begin any time
                      after the submission.
     pauseScope(): Completes the execution of all started tensor operations in the
                   current DAG and defers the execution of the rest of the DAG for later.
     resumeScope(name): Resumes the execution of a previously paused DAG, making it current.
     closeScope(): Completes all tensor operations in the current DAG and destroys it.
 (c) submit(TensorOperation): Submits a tensor operation for (generally deferred) execution.
     sync(TensorOperation): Tests the completion of a specific tensor operation.
     sync(tensor): Tests the completion of all submitted update operations on a given tensor.
**/

#ifndef EXATN_RUNTIME_TENSORRUNTIME_HPP_
#define EXATN_RUNTIME_TENSORRUNTIME_HPP_

#include <iostream>
#include <memory>
#include <string>
#include <mutex>

#include "TensorGraph.hpp"
#include "tensor.hpp"
#include "tensor_operation.hpp"
#include "tensor_method.hpp"

namespace exatn {
namespace runtime {

class TensorRuntime {

protected:

  /** Active execution graphs (DAGs) **/
  std::map<std::string, std::shared_ptr<TensorGraph>> dags;
  /** Table for tracking the execution status on a given tensor:
  Scope name --> Scope map: Tensor Id --> Number of outstanding update operations **/
  std::map<std::string, std::map<std::size_t, int>> outTensorExecTbl;
  /** Name of the current scope (current DAG) **/
  std::string currentScope;
  /** Mutex for locking outTensorExec and dags **/
  std::mutex mtx;

public:

  TensorRuntime() = default;
  TensorRuntime(const TensorRuntime &) = delete;
  TensorRuntime & operator=(const TensorRuntime &) = delete;
  TensorRuntime(TensorRuntime &&) noexcept = default;
  TensorRuntime & operator=(TensorRuntime &&) noexcept = default;
  ~TensorRuntime() = default;

  /** Opens a new scope represented by a new execution graph. **/
  void openScope(const std::string & scopeName);

  /** Pauses the current scope by completing all outstanding tensor operations
      and pausing the further progress of the current execution graph until resume. **/
  void pauseScope();

  /** Resumes the execution of a previously paused scope (execution graph). **/
  void resumeScope(const std::string & scopeName);

  /** Closes the current scope, fully completing the current execution graph. **/
  void closeScope();

  /** Submits a tensor operation into the current execution graph. **/
  void submit(std::shared_ptr<numerics::TensorOperation> op);

  /** Tests for completion of a given tensor operation.
      If wait = TRUE, it will block until completion. **/
  bool sync(const numerics::TensorOperation & op,
            bool wait = false);

  /** Tests for completion of all outstanding update operations on a given tensor.
      If wait = TRUE, it will block until completion. **/
  bool sync(const numerics::Tensor & tensor,
            bool wait = false);

  /** Returns an accessor to the elements of a given tensor. **/
  TensorDenseBlock getTensorData(const numerics::Tensor & tensor);
};

} // namespace runtime
} // namespace exatn

#endif //EXATN_RUNTIME_TENSORRUNTIME_HPP_
