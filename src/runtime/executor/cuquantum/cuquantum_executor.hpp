/** ExaTN: Tensor Runtime: Tensor network executor: NVIDIA cuQuantum
REVISION: 2021/12/27

Copyright (C) 2018-2021 Dmitry Lyakh
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle)

Rationale:
 - ExaTN graph executor may accept whole tensor networks for execution
   via the optional cuQuantum backend in which case the graph executor
   will delegate execution of whole tensor networks to CuQuantumExecutor.

**/

#ifdef CUQUANTUM

#ifndef EXATN_RUNTIME_CUQUANTUM_EXECUTOR_HPP_
#define EXATN_RUNTIME_CUQUANTUM_EXECUTOR_HPP_

#include <unordered_map>
#include <vector>
#include <functional>

#include "tensor_network_queue.hpp"

namespace talsh{
class Tensor;
}

namespace exatn {
namespace runtime {

using TensorImplFunc = std::function<const void*(const numerics::Tensor &, int, int, std::size_t *)>;
using TensorImplTalshFunc = std::function<std::shared_ptr<talsh::Tensor>(const numerics::Tensor &, int, int)>;

struct TensorNetworkReq;


class CuQuantumExecutor {

public:

 CuQuantumExecutor(TensorImplFunc tensor_data_access_func);

 CuQuantumExecutor(const CuQuantumExecutor &) = delete;
 CuQuantumExecutor & operator=(CuQuantumExecutor &) = delete;
 CuQuantumExecutor(CuQuantumExecutor &&) noexcept = delete;
 CuQuantumExecutor & operator=(CuQuantumExecutor &&) noexcept = delete;
 virtual ~CuQuantumExecutor();

 /** Submits a tensor network for execution via CuQuantumExecutor.
     The associated tensor network execution handle can be used
     for progressing and completing the tensor network execution. **/
 TensorNetworkQueue::ExecStat execute(std::shared_ptr<numerics::TensorNetwork> network,
                                      const TensorOpExecHandle exec_handle);

 /** Synchronizes on the progress of the tensor network execution.
     If wait = TRUE, waits until completion, otherwise just tests the progress.
     Returns the current status of the tensor network execution. **/
 TensorNetworkQueue::ExecStat sync(const TensorOpExecHandle exec_handle,
                                   int * error_code,
                                   bool wait = true);

 /** Synchronizes execution of all submitted tensor networks to completion. **/
 bool sync();

protected:

 struct DeviceAttr{
  void * buffer_ptr = nullptr;
  std::size_t buffer_size = 0;
  void * workspace_ptr = nullptr;
  std::size_t workspace_size = 0;
  void * cutn_handle; //cutensornetHandle_t = void*
 };

 /** Currently processed tensor networks **/
 std::unordered_map<TensorOpExecHandle,std::shared_ptr<TensorNetworkReq>> active_networks_;
 /** Attributes of all GPUs available to the current process **/
 std::vector<std::pair<int,DeviceAttr>> gpu_attr_; //{gpu_id, gpu_attributes}
 /** Tensor data access function **/
 TensorImplFunc tensor_data_access_func_; //numerics::Tensor --> {tensor_body_ptr, size_in_bytes}
};

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_CUQUANTUM_EXECUTOR_HPP_

#endif //CUQUANTUM
