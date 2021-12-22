/** ExaTN: Tensor Runtime: Tensor network executor: NVIDIA cuQuantum
REVISION: 2021/12/22

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

#include "tensor_network_queue.hpp"

namespace exatn {
namespace runtime {

struct TensorNetworkReq;

class CuQuantumExecutor {

public:

 CuQuantumExecutor();
 CuQuantumExecutor(const CuQuantumExecutor &) = delete;
 CuQuantumExecutor & operator=(CuQuantumExecutor &) = delete;
 CuQuantumExecutor(CuQuantumExecutor &&) noexcept = delete;
 CuQuantumExecutor & operator=(CuQuantumExecutor &&) noexcept = delete;
 virtual ~CuQuantumExecutor();

 int execute(std::shared_ptr<numerics::TensorNetwork> network,
             TensorOpExecHandle exec_handle);

 bool sync(TensorOpExecHandle exec_handle,
           int * error_code,
           bool wait = true);

 bool sync();

protected:

 /** Currently processed tensor networks **/
 std::unordered_map<TensorOpExecHandle,std::shared_ptr<TensorNetworkReq>> active_networks_;
 /** GPU Ids available to the current process **/
 std::vector<int> gpus;
 /** cuTensorNet contexts for all available GPUs **/
 std::vector<void*> ctn_handles; //cutensornetHandle_t
};

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_CUQUANTUM_EXECUTOR_HPP_

#endif //CUQUANTUM
