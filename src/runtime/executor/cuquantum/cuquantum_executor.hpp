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

 int execute(std::shared_ptr<numerics::TensorNetwork> network,
             const TensorOpExecHandle exec_handle);

 bool executing(const TensorOpExecHandle exec_handle);

 bool sync(const TensorOpExecHandle exec_handle,
           int * error_code,
           bool wait = true);

 bool sync();

protected:

 /** Currently processed tensor networks **/
 std::unordered_map<TensorOpExecHandle,std::shared_ptr<TensorNetworkReq>> active_networks_;
 /** GPU Ids available to the current process **/
 std::vector<int> gpus_;
 /** cuTensorNet contexts for all available GPUs **/
 std::vector<void*> ctn_handles_; //cutensornetHandle_t = void*
 /** Tensor data access function **/
 TensorImplFunc tensor_data_access_func_;
};

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_CUQUANTUM_EXECUTOR_HPP_

#endif //CUQUANTUM
