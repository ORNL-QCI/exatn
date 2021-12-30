/** ExaTN: Tensor Runtime: Tensor network executor: NVIDIA cuQuantum
REVISION: 2021/12/30

Copyright (C) 2018-2021 Dmitry Lyakh
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle)

Rationale:

**/

#ifdef CUQUANTUM

#include <cutensornet.h>
#include <cutensor.h>
#include <cuda_runtime.h>

#include <vector>
#include <unordered_map>
#include <type_traits>

#include <iostream>

#include "talshxx.hpp"

#include "cuquantum_executor.hpp"


#define HANDLE_CUDA_ERROR(x) \
{ const auto err = x; \
  if( err != cudaSuccess ) \
{ printf("Error: %s in line %d\n", cudaGetErrorString(err), __LINE__); std::abort(); } \
};

#define HANDLE_CTN_ERROR(x) \
{ const auto err = x; \
  if( err != CUTENSORNET_STATUS_SUCCESS ) \
{ printf("Error: %s in line %d\n", cutensornetGetErrorString(err), __LINE__); std::abort(); } \
};


namespace exatn {
namespace runtime {

struct TensorDescriptor {
 std::vector<int64_t> extents; //tensor dimension extents
 std::vector<int64_t> strides; //tensor dimension strides (optional)
 void * body_ptr = nullptr;    //pointer to the tensor body image
 std::size_t volume = 0;       //tensor body volume
 cudaDataType_t data_type;     //tensor element data type
};

struct TensorNetworkReq {
 TensorNetworkQueue::ExecStat exec_status = TensorNetworkQueue::ExecStat::None; //tensor network execution status
 std::shared_ptr<numerics::TensorNetwork> network; //tensor network specification
 std::unordered_map<numerics::TensorHashType,TensorDescriptor> tensor_descriptors; //tensor descriptors (shape, volume, data type, body)
 std::unordered_map<unsigned int, std::vector<int32_t>> tensor_modes; //indices associated with tensor dimensions (key is the original tensor id)
 std::unordered_map<int32_t,int64_t> index_extents; //extent of each registered tensor mode
 std::vector<void*> memory_window_ptr; //end of the GPU memory segment allocated for the tensors
 cutensornetNetworkDescriptor_t net_descriptor;
 cutensornetContractionOptimizerConfig_t opt_config;
 cutensornetContractionOptimizerInfo_t opt_info;
 cutensornetContractionPlan_t comp_plan;
 cudaStream_t stream;
 cutensornetComputeType_t compute_type;
};


CuQuantumExecutor::CuQuantumExecutor(TensorImplFunc tensor_data_access_func):
 tensor_data_access_func_(std::move(tensor_data_access_func))
{
 static_assert(std::is_same<cutensornetHandle_t,void*>::value,"#FATAL(exatn::runtime::CuQuantumExecutor): cutensornetHandle_t != (void*)");

 const size_t version = cutensornetGetVersion();
 std::cout << "#DEBUG(exatn::runtime::CuQuantumExecutor): cuTensorNet backend version " << version << std::endl;

 int num_gpus = 0;
 auto error_code = talshDeviceCount(DEV_NVIDIA_GPU,&num_gpus); assert(error_code == TALSH_SUCCESS);
 for(int i = 0; i < num_gpus; ++i){
  if(talshDeviceState(i,DEV_NVIDIA_GPU) >= DEV_ON){
   gpu_attr_.emplace_back(std::make_pair(i,DeviceAttr{}));
   gpu_attr_.back().second.workspace_ptr = talsh::getDeviceBufferBasePtr(DEV_NVIDIA_GPU,i);
   assert(reinterpret_cast<std::size_t>(gpu_attr_.back().second.workspace_ptr) % MEM_ALIGNMENT == 0);
   gpu_attr_.back().second.buffer_size = talsh::getDeviceMaxBufferSize(DEV_NVIDIA_GPU,i);
   std::size_t wrk_size = (std::size_t)(static_cast<float>(gpu_attr_.back().second.buffer_size) * WORKSPACE_FRACTION);
   wrk_size -= wrk_size % MEM_ALIGNMENT;
   gpu_attr_.back().second.workspace_size = wrk_size;
   gpu_attr_.back().second.buffer_size -= wrk_size;
   gpu_attr_.back().second.buffer_size -= gpu_attr_.back().second.buffer_size % MEM_ALIGNMENT;
   gpu_attr_.back().second.buffer_ptr = (void*)(((char*)(gpu_attr_.back().second.workspace_ptr)) + wrk_size);
   mem_pool_.emplace_back(LinearMemoryPool(gpu_attr_.back().second.buffer_ptr,
                                           gpu_attr_.back().second.buffer_size,MEM_ALIGNMENT));
  }
 }
 std::cout << "#DEBUG(exatn::runtime::CuQuantumExecutor): Number of available GPUs = " << gpu_attr_.size() << std::endl;

 for(const auto & gpu: gpu_attr_){
  HANDLE_CUDA_ERROR(cudaSetDevice(gpu.first));
  HANDLE_CTN_ERROR(cutensornetCreate((cutensornetHandle_t*)(&gpu.second.cutn_handle)));
 }
 std::cout << "#DEBUG(exatn::runtime::CuQuantumExecutor): Created cuTensorNet contexts for all available GPUs" << std::endl;

 std::cout << "#DEBUG(exatn::runtime::CuQuantumExecutor): GPU configuration:\n";
 for(const auto & gpu: gpu_attr_){
  std::cout << " GPU #" << gpu.first
            << ": wrk_ptr = " << gpu.second.workspace_ptr
            << ", size = " << gpu.second.workspace_size
            << "; buf_ptr = " << gpu.second.buffer_ptr
            << ", size = " << gpu.second.buffer_size << std::endl;
 }
}


CuQuantumExecutor::~CuQuantumExecutor()
{
 sync();
 for(const auto & gpu: gpu_attr_){
  HANDLE_CUDA_ERROR(cudaSetDevice(gpu.first));
  HANDLE_CTN_ERROR(cutensornetDestroy((cutensornetHandle_t)(gpu.second.cutn_handle)));
 }
 std::cout << "#DEBUG(exatn::runtime::CuQuantumExecutor): Destroyed cuTensorNet contexts for all available GPUs" << std::endl;
 gpu_attr_.clear();
}


TensorNetworkQueue::ExecStat CuQuantumExecutor::execute(std::shared_ptr<numerics::TensorNetwork> network,
                                                        const TensorOpExecHandle exec_handle)
{
 assert(network);
 TensorNetworkQueue::ExecStat exec_stat = TensorNetworkQueue::ExecStat::None;
 auto res = active_networks_.emplace(std::make_pair(exec_handle, new TensorNetworkReq{}));
 if(res.second){
  auto tn_req = res.first->second;
  tn_req->network = network;
  tn_req->exec_status = TensorNetworkQueue::ExecStat::Idle;
  parseTensorNetwork(tn_req); //still Idle
  loadTensors(tn_req); //Idle --> Loading
  if(tn_req->exec_status == TensorNetworkQueue::ExecStat::Loading){
   planExecution(tn_req); //Loading --> Planning (while loading data)
   if(tn_req->exec_status == TensorNetworkQueue::ExecStat::Planning){
    contractTensorNetwork(tn_req); //Planning --> Executing
   }
  }
  exec_stat = tn_req->exec_status;
 }else{
  std::cout << "#WARNING(exatn::runtime::CuQuantumExecutor): execute: Repeated tensor network submission detected!\n";
 }
 return exec_stat;
}


TensorNetworkQueue::ExecStat CuQuantumExecutor::sync(const TensorOpExecHandle exec_handle,
                                                     int * error_code)
{
 *error_code = 0;
 TensorNetworkQueue::ExecStat exec_stat = TensorNetworkQueue::ExecStat::None;
 auto iter = active_networks_.find(exec_handle);
 if(iter != active_networks_.end()){
  auto tn_req = iter->second;
  if(tn_req->exec_status == TensorNetworkQueue::ExecStat::Executing){
   testCompletion(tn_req); //Executing --> Completed
  }else{
   if(tn_req->exec_status == TensorNetworkQueue::ExecStat::Idle)
    loadTensors(tn_req); //Idle --> Loading
   if(tn_req->exec_status == TensorNetworkQueue::ExecStat::Loading)
    planExecution(tn_req); //Loading --> Planning (while loading data)
   if(tn_req->exec_status == TensorNetworkQueue::ExecStat::Planning)
    contractTensorNetwork(tn_req); //Planning --> Executing
  }
  exec_stat = tn_req->exec_status;
  tn_req.reset();
  if(exec_stat == TensorNetworkQueue::ExecStat::Completed)
   active_networks_.erase(iter);
 }
 return exec_stat;
}


void CuQuantumExecutor::sync()
{
 while(!active_networks_.empty()){
  for(auto iter = active_networks_.begin(); iter != active_networks_.end(); ++iter){
   int error_code = 0;
   const auto exec_stat = sync(iter->first,&error_code); assert(error_code == 0);
   if(exec_stat == TensorNetworkQueue::ExecStat::Completed) break;
  }
 }
 return;
}


void CuQuantumExecutor::parseTensorNetwork(std::shared_ptr<TensorNetworkReq> tn_req)
{
 
 return;
}


void CuQuantumExecutor::loadTensors(std::shared_ptr<TensorNetworkReq> tn_req)
{
 
 return;
}


void CuQuantumExecutor::planExecution(std::shared_ptr<TensorNetworkReq> tn_req)
{
 
 return;
}


void CuQuantumExecutor::contractTensorNetwork(std::shared_ptr<TensorNetworkReq> tn_req)
{
 
 return;
}


void CuQuantumExecutor::testCompletion(std::shared_ptr<TensorNetworkReq> tn_req)
{
 
 return;
}

} //namespace runtime
} //namespace exatn

#endif //CUQUANTUM
