/** ExaTN: Tensor Runtime: Tensor network executor: NVIDIA cuQuantum
REVISION: 2022/12/19

Copyright (C) 2018-2022 Dmitry Lyakh
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle)
Copyright (C) 2022-2022 NVIDIA Corporation

SPDX-License-Identifier: BSD-3-Clause

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

#include "linear_memory.hpp"
#include "tensor_network_queue.hpp"
#include "mpi_proxy.hpp"

#include "errors.hpp"


namespace talsh{
class Tensor;
}

namespace exatn {

namespace runtime {

using TensorImplFunc = std::function<void*(const numerics::Tensor &, int, int, std::size_t *)>;
using TensorImplTalshFunc = std::function<std::shared_ptr<talsh::Tensor>(const numerics::Tensor &, int, int)>;

struct TensorNetworkReq;

struct ExecutionTimings {
 float prepare = 0.0;
 float data_in = 0.0;
 float data_out = 0.0;
 float compute = 0.0;

 static ExecutionTimings computeAverage(const std::vector<ExecutionTimings> & timings);
 static ExecutionTimings computeWorst(const std::vector<ExecutionTimings> & timings);
 static ExecutionTimings computeBest(const std::vector<ExecutionTimings> & timings);
};


class CuQuantumExecutor {

public:

 static constexpr unsigned int MIN_HYPER_SAMPLES = 32; // > 0
 static constexpr unsigned int RECONFIG_ITERATIONS = 256;
 static constexpr unsigned int RECONFIG_LEAVES = 6;

 static_assert(MIN_HYPER_SAMPLES > 0, "#FATAL(exatn::runtime::cuquantum): MIN_HYPER_SAMPLES must be greater than zero!");

 CuQuantumExecutor(TensorImplFunc tensor_data_access_func,
                   unsigned int pipeline_depth,
                   unsigned int num_processes,
                   unsigned int process_rank);

 CuQuantumExecutor(const CuQuantumExecutor &) = delete;
 CuQuantumExecutor & operator=(CuQuantumExecutor &) = delete;
 CuQuantumExecutor(CuQuantumExecutor &&) noexcept = delete;
 CuQuantumExecutor & operator=(CuQuantumExecutor &&) noexcept = delete;
 virtual ~CuQuantumExecutor();

 /** Submits a tensor network for execution via CuQuantumExecutor.
     The associated tensor network execution handle can be used
     for progressing and completing the tensor network execution. **/
#ifdef MPI_ENABLED
 TensorNetworkQueue::ExecStat execute(std::shared_ptr<numerics::TensorNetwork> network, //in: tensor network
                                      unsigned int num_processes, //in: total number of executing processes
                                      unsigned int process_rank,  //in: rank of the current executing process
                                      const MPICommProxy & communicator, //in: MPI communicator
                                      const TensorOpExecHandle exec_handle); //in: tensor network execution handle
#else
 TensorNetworkQueue::ExecStat execute(std::shared_ptr<numerics::TensorNetwork> network, //in: tensor network
                                      unsigned int num_processes, //in: total number of executing processes
                                      unsigned int process_rank,  //in: rank of the current executing process
                                      const TensorOpExecHandle exec_handle); //in: tensor network execution handle
#endif

 /** Synchronizes on the progress of the tensor network execution.
     If wait = TRUE, waits until completion, otherwise just tests the progress.
     Returns the current status of the tensor network execution. **/
 TensorNetworkQueue::ExecStat sync(const TensorOpExecHandle exec_handle, //in: tensor network execution handle
                                   int * error_code, //out: error code (0:success)
                                   int64_t * num_slices = nullptr, //out: number of tensor network slices
                                   std::vector<ExecutionTimings> * timings = nullptr); //out: execution timings (ms)

 /** Synchronizes execution of all submitted tensor networks to completion. **/
 void sync();

 /** Returns the total executed flop count. **/
 double getTotalFlopCount() const {return flops_;}

protected:

 static constexpr float WORKSPACE_FRACTION = 0.6;
 static constexpr std::size_t MEM_ALIGNMENT = 256;

 void acquireWorkspace(unsigned int dev,
                       void ** workspace_ptr,
                       uint64_t * workspace_size);

 void parseTensorNetwork(std::shared_ptr<TensorNetworkReq> tn_req);
 void loadTensors(std::shared_ptr<TensorNetworkReq> tn_req);
 void planExecution(std::shared_ptr<TensorNetworkReq> tn_req);
 void contractTensorNetwork(std::shared_ptr<TensorNetworkReq> tn_req);
 void testCompletion(std::shared_ptr<TensorNetworkReq> tn_req);

 struct DeviceAttr{
  void * buffer_ptr = nullptr;
  std::size_t buffer_size = 0;
  void * workspace_ptr = nullptr;
  std::size_t workspace_size = 0;
  unsigned int pipe_level = 0;
  void * cutn_handle; //cutensornetHandle_t = void*
 };

 /** Currently processed (progressing) tensor networks **/
 std::unordered_map<TensorOpExecHandle,std::shared_ptr<TensorNetworkReq>> active_networks_;
 /** Attributes of all GPUs available to the current process **/
 std::vector<std::pair<int,DeviceAttr>> gpu_attr_; //{gpu_id, gpu_attributes}
 /** Moving-window linear memory pools for all GPUs of the current process **/
 std::vector<LinearMemoryPool> mem_pool_;
 /** Tensor data access function **/
 TensorImplFunc tensor_data_access_func_; //numerics::Tensor --> {tensor_body_ptr, size_in_bytes}
 /** Pipeline depth **/
 const unsigned int pipe_depth_;
 /** Total number of parallel processes **/
 const unsigned int num_processes_;
 /** Current process rank **/
 const unsigned int process_rank_;
 /** Executed flops **/
 double flops_;
};

} //namespace runtime
} //namespace exatn

#endif //EXATN_RUNTIME_CUQUANTUM_EXECUTOR_HPP_

#endif //CUQUANTUM
