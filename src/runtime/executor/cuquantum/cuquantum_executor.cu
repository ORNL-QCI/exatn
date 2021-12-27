/** ExaTN: Tensor Runtime: Tensor network executor: NVIDIA cuQuantum
REVISION: 2021/12/27

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
 std::vector<int32_t> modes;
 std::vector<int64_t> extents;
};

struct TensorNetworkReq {
 std::shared_ptr<numerics::TensorNetwork> network;
 std::unordered_map<numerics::TensorHashType,TensorDescriptor> tensor_descriptors;
 std::unordered_map<int32_t,int64_t> index_extents;
 cutensornetNetworkDescriptor_t net_descriptor;
 cutensornetContractionOptimizerConfig_t opt_config;
 cutensornetContractionOptimizerInfo_t opt_info;
 cutensornetContractionPlan_t comp_plan;
 cudaStream_t stream;
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
  if(talshDeviceState(i,DEV_NVIDIA_GPU) >= DEV_ON) gpus_.emplace_back(i);
 }
 std::cout << "#DEBUG(exatn::runtime::CuQuantumExecutor): Number of available GPUs = " << gpus_.size() << std::endl;

 ctn_handles_.resize(gpus_.size());
 for(const auto & gpu_id: gpus_){
  HANDLE_CUDA_ERROR(cudaSetDevice(gpu_id));
  HANDLE_CTN_ERROR(cutensornetCreate((cutensornetHandle_t*)(&ctn_handles_[gpu_id])));
 }
 std::cout << "#DEBUG(exatn::runtime::CuQuantumExecutor): Created cuTensorNet contexts for all available GPUs" << std::endl;
}


CuQuantumExecutor::~CuQuantumExecutor()
{
 bool success = sync(); assert(success);
 for(const auto & gpu_id: gpus_){
  HANDLE_CUDA_ERROR(cudaSetDevice(gpu_id));
  HANDLE_CTN_ERROR(cutensornetDestroy((cutensornetHandle_t)(ctn_handles_[gpu_id])));
 }
 std::cout << "#DEBUG(exatn::runtime::CuQuantumExecutor): Destroyed cuTensorNet contexts for all available GPUs" << std::endl;
 ctn_handles_.clear();
 gpus_.clear();
}


int CuQuantumExecutor::execute(std::shared_ptr<numerics::TensorNetwork> network,
                               const TensorOpExecHandle exec_handle)
{
 int error_code = 0;
 //`Finish
 return error_code;
}


bool CuQuantumExecutor::executing(const TensorOpExecHandle exec_handle)
{
 auto iter = active_networks_.find(exec_handle);
 return (iter != active_networks_.end());
}


bool CuQuantumExecutor::sync(const TensorOpExecHandle exec_handle,
                             int * error_code,
                             bool wait)
{
 bool synced = true;
 *error_code = 0;
 auto iter = active_networks_.find(exec_handle);
 if(iter != active_networks_.end()){
  //`Finish
 }
 return synced;
}


bool CuQuantumExecutor::sync()
{
 bool synced = true;
 //`Finish
 return synced;
}

} //namespace runtime
} //namespace exatn

#endif //CUQUANTUM
