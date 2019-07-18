#ifndef EXATN_RUNTIME_TENSORRUNTIME_HPP_
#define EXATN_RUNTIME_TENSORRUNTIME_HPP_

#include <iostream>
#include <memory>
#include <mutex>

#include "TensorGraph.hpp"
#include "tensor.hpp"
#include "tensor_operation.hpp"
#include "tensor_method.hpp"

namespace exatn {
namespace runtime {

class TensorRuntime {

protected:
  std::map<std::string, std::shared_ptr<TensorGraph>> dags; //execution graphs
  std::string currentScope; //name of the current scope
  std::map<std::string, std::map<std::size_t, int>> outTensorExecTbl; //table for tracking output tensor execution
  std::mutex mtx; //mutex for locking outTensorExec and dags

public:

  /** Opens a new scope represented by a new execution graph. **/
  void openScope(const std::string & scopeName);

  /** Pauses the current scope by completing all outstanding tensor operations
      and pausing the further progress of the current execution graph until resume. **/
  void pauseScope();

  /** Resumes the execution of the previously paused scope. **/
  void resumeScope(const std::string & scopeName);

  /** Closes the current scope, fully completing the current execution graph. **/
  void closeScope();

  /** Submits a tensor operation into the current execution graph. **/
  void submit(std::shared_ptr<numerics::TensorOperation> op);

  /** Tests for completion of a given tensor operation.
      If wait = TRUE, it will block until completion. **/
  bool sync(const numerics::TensorOperation & op,
            bool wait = false);

  /** Tests for completion of all outstanding tensor operations on a given tensor.
      If wait = TRUE, it will block until completion. **/
  bool sync(const numerics::Tensor & tensor,
            bool wait = false);

  /** Returns an accessor to the elements of a given tensor. **/
  TensorDenseBlock getTensorData(const numerics::Tensor & tensor);
};

} // namespace runtime
} // namespace exatn

#endif //EXATN_RUNTIME_TENSORRUNTIME_HPP_
