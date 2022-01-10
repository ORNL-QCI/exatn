/** ExaTN: Tensor Runtime: Tensor network executor: NVIDIA cuQuantum
REVISION: 2022/01/10

Copyright (C) 2018-2022 Dmitry Lyakh
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle)

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
#include "timers.hpp"

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
 cudaDataType_t data_type;     //tensor element data type
 std::size_t volume = 0;       //tensor body volume
 std::size_t size = 0;         //tensor body size (bytes)
 void * src_ptr = nullptr;     //non-owning pointer to the tensor body source image
 std::vector<void*> dst_ptr;   //non-owning pointer to the tensor body destination image (on each GPU)
};

struct TensorNetworkReq {
 TensorNetworkQueue::ExecStat exec_status = TensorNetworkQueue::ExecStat::None; //tensor network execution status
 int num_procs = 0; //total number of executing processes
 int proc_id = -1; //id of the current executing process
 int64_t num_slices = 0;
 std::shared_ptr<numerics::TensorNetwork> network; //tensor network specification
 std::unordered_map<numerics::TensorHashType, TensorDescriptor> tensor_descriptors; //tensor descriptors (shape, volume, data type, body)
 std::unordered_map<unsigned int, std::vector<int32_t>> tensor_modes; //indices associated with tensor dimensions (key = original tensor id)
 std::unordered_map<int32_t, int64_t> mode_extents; //extent of each registered tensor mode
 int32_t * num_modes_in = nullptr;
 int64_t ** extents_in = nullptr; //non-owning
 int64_t ** strides_in = nullptr;
 int32_t ** modes_in = nullptr; //non-owning
 uint32_t * alignments_in = nullptr;
 void ** data_in = nullptr;
 int32_t num_modes_out;
 int64_t * extents_out = nullptr; //non-owning
 int64_t * strides_out = nullptr;
 int32_t * modes_out = nullptr; //non-owning
 uint32_t alignment_out;
 void * data_out = nullptr; //non-owning
 void * workspace = nullptr; //non-owning
 uint64_t worksize = 0;
 std::vector<void*> memory_window_ptr; //end of the GPU memory segment allocated for the tensors (on each GPU)
 cutensornetNetworkDescriptor_t net_descriptor;
 cutensornetContractionOptimizerConfig_t opt_config;
 cutensornetContractionOptimizerInfo_t opt_info;
 cutensornetContractionPlan_t comp_plan;
 cudaDataType_t data_type;
 cutensornetComputeType_t compute_type;
 cudaStream_t stream;
 cudaEvent_t data_in_start;
 cudaEvent_t data_in_finish;
 cudaEvent_t compute_start;
 cudaEvent_t compute_finish;
 cudaEvent_t data_out_finish;
 double prepare_start;
 double prepare_finish;

 ~TensorNetworkReq() {
  cudaStreamSynchronize(stream);
  cudaEventDestroy(data_out_finish);
  cudaEventDestroy(compute_finish);
  cudaEventDestroy(compute_start);
  cudaEventDestroy(data_in_finish);
  cudaEventDestroy(data_in_start);
  cudaStreamDestroy(stream);
  cutensornetDestroyContractionPlan(comp_plan);
  cutensornetDestroyContractionOptimizerConfig(opt_config);
  cutensornetDestroyContractionOptimizerInfo(opt_info);
  cutensornetDestroyNetworkDescriptor(net_descriptor);
  //if(modes_out != nullptr) delete [] modes_out;
  if(strides_out != nullptr) delete [] strides_out;
  //if(extents_out != nullptr) delete [] extents_out;
  if(data_in != nullptr) delete [] data_in;
  if(alignments_in != nullptr) delete [] alignments_in;
  //if(modes_in != nullptr) delete [] modes_in;
  if(strides_in != nullptr) delete [] strides_in;
  //if(extents_in != nullptr) delete [] extents_in;
  if(num_modes_in != nullptr) delete [] num_modes_in;
 }
};


CuQuantumExecutor::CuQuantumExecutor(TensorImplFunc tensor_data_access_func,
                                     unsigned int pipeline_depth,
                                     unsigned int num_processes, unsigned int process_rank):
 tensor_data_access_func_(std::move(tensor_data_access_func)),
 pipe_depth_(pipeline_depth), num_processes_(num_processes), process_rank_(process_rank), flops_(0.0)
{
 static_assert(std::is_same<cutensornetHandle_t,void*>::value,"#FATAL(exatn::runtime::CuQuantumExecutor): cutensornetHandle_t != (void*)");

 const size_t version = cutensornetGetVersion();
 std::cout << "#DEBUG(exatn::runtime::CuQuantumExecutor): cuTensorNet backend version " << version << std::endl;

 std::cout << "#DEBUG(exatn::runtime::CuQuantumExecutor): Total number of processes = " << num_processes_ << std::endl;
 int num_gpus = 0;
 auto error_code = talshDeviceCount(DEV_NVIDIA_GPU,&num_gpus); assert(error_code == TALSH_SUCCESS);
 for(int i = 0; i < num_gpus; ++i){
  if(talshDeviceState(i,DEV_NVIDIA_GPU) >= DEV_ON){
   gpu_attr_.emplace_back(std::make_pair(i,DeviceAttr{}));
   gpu_attr_.back().second.pipe_level = 0;
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
 std::cout << "#MSG(exatn::cuQuantum): Statistics across all GPU devices:\n";
 std::cout << " Number of Flops processed: " << flops_ << std::endl;
 std::cout << "#END_MSG\n";
 gpu_attr_.clear();
}


TensorNetworkQueue::ExecStat CuQuantumExecutor::execute(std::shared_ptr<numerics::TensorNetwork> network,
                                                        unsigned int num_processes, unsigned int process_rank,
                                                        const TensorOpExecHandle exec_handle)
{
 assert(network);
 TensorNetworkQueue::ExecStat exec_stat = TensorNetworkQueue::ExecStat::None;
 auto res = active_networks_.emplace(std::make_pair(exec_handle, new TensorNetworkReq{}));
 if(res.second){
  auto tn_req = res.first->second;
  tn_req->network = network;
  tn_req->exec_status = TensorNetworkQueue::ExecStat::Idle;
  tn_req->num_procs = num_processes;
  tn_req->proc_id = process_rank;
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
                                                     int * error_code,
                                                     int64_t * num_slices,
                                                     ExecutionTimings * timings)
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
  if(exec_stat == TensorNetworkQueue::ExecStat::Completed){
   if(num_slices != nullptr) *num_slices = tn_req->num_slices;
   if(timings != nullptr){
    timings->prepare = (tn_req->prepare_finish - tn_req->prepare_start) * 1000.0;
    HANDLE_CUDA_ERROR(cudaEventElapsedTime(&(timings->data_in),tn_req->data_in_start,tn_req->data_in_finish));
    HANDLE_CUDA_ERROR(cudaEventElapsedTime(&(timings->data_out),tn_req->compute_finish,tn_req->data_out_finish));
    HANDLE_CUDA_ERROR(cudaEventElapsedTime(&(timings->compute),tn_req->compute_start,tn_req->compute_finish));
   }
  }
  tn_req.reset();
  if(exec_stat == TensorNetworkQueue::ExecStat::Completed) active_networks_.erase(iter);
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


cudaDataType_t getCudaDataType(const TensorElementType elem_type)
{
 cudaDataType_t cuda_data_type;
 switch(elem_type){
 case TensorElementType::REAL32: cuda_data_type = CUDA_R_32F; break;
 case TensorElementType::REAL64: cuda_data_type = CUDA_R_64F; break;
 case TensorElementType::COMPLEX32: cuda_data_type = CUDA_C_32F; break;
 case TensorElementType::COMPLEX64: cuda_data_type = CUDA_C_64F; break;
 default:
  assert(false);
 }
 return cuda_data_type;
}


cutensornetComputeType_t getCutensorComputeType(const TensorElementType elem_type)
{
 cutensornetComputeType_t cutensor_data_type;
 switch(elem_type){
 case TensorElementType::REAL32: cutensor_data_type = CUTENSORNET_COMPUTE_32F; break;
 case TensorElementType::REAL64: cutensor_data_type = CUTENSORNET_COMPUTE_64F; break;
 case TensorElementType::COMPLEX32: cutensor_data_type = CUTENSORNET_COMPUTE_32F; break;
 case TensorElementType::COMPLEX64: cutensor_data_type = CUTENSORNET_COMPUTE_64F; break;
 default:
  assert(false);
 }
 return cutensor_data_type;
}


void CuQuantumExecutor::acquireWorkspace(unsigned int dev,
                                         void ** workspace_ptr,
                                         uint64_t * workspace_size)
{
 assert(dev < gpu_attr_.size());
 auto & dev_attr = gpu_attr_[dev].second;
 *workspace_size = dev_attr.workspace_size / pipe_depth_;
 *workspace_ptr = (void*)((char*)(dev_attr.workspace_ptr) + ((*workspace_size) * dev_attr.pipe_level));
 dev_attr.pipe_level = (++(dev_attr.pipe_level)) % pipe_depth_;
 return;
}


void CuQuantumExecutor::parseTensorNetwork(std::shared_ptr<TensorNetworkReq> tn_req)
{
 const auto & net = *(tn_req->network);
 const int32_t num_input_tensors = net.getNumTensors();
 tn_req->num_modes_in = new int32_t[num_input_tensors];
 tn_req->extents_in = new int64_t*[num_input_tensors];
 tn_req->strides_in = new int64_t*[num_input_tensors];
 tn_req->modes_in = new int32_t*[num_input_tensors];
 tn_req->alignments_in = new uint32_t[num_input_tensors];
 tn_req->data_in = new void*[num_input_tensors];

 for(unsigned int i = 0; i < num_input_tensors; ++i) tn_req->strides_in[i] = NULL;
 for(unsigned int i = 0; i < num_input_tensors; ++i) tn_req->alignments_in[i] = MEM_ALIGNMENT;
 tn_req->strides_out = NULL;
 tn_req->alignment_out = MEM_ALIGNMENT;

 int32_t mode_id = 0, tens_num = 0;
 for(auto iter = net.cbegin(); iter != net.cend(); ++iter){
  const auto tens_id = iter->first;
  const auto & tens = iter->second;
  const auto tens_hash = tens.getTensor()->getTensorHash();
  const auto tens_vol = tens.getTensor()->getVolume();
  const auto tens_rank = tens.getRank();
  const auto tens_type = tens.getElementType();
  if(tens_type == TensorElementType::VOID){
   std::cout << "#ERROR(exatn::runtime::CuQuantumExecutor): Network tensor #" << tens_id
             << " has not been allocated typed storage yet!\n";
   assert(false);
  }
  const auto & tens_legs = tens.getTensorLegs();
  const auto & tens_dims = tens.getDimExtents();

  auto res0 = tn_req->tensor_descriptors.emplace(std::make_pair(tens_hash,TensorDescriptor{}));
  if(res0.second){
   auto & descr = res0.first->second;
   descr.extents.resize(tens_rank);
   for(unsigned int i = 0; i < tens_rank; ++i) descr.extents[i] = tens_dims[i];
   descr.data_type = getCudaDataType(tens_type);
   descr.volume = tens_vol;
   descr.src_ptr = tensor_data_access_func_(*(tens.getTensor()),DEV_HOST,0,&(descr.size)); //`Assuming tensor body is on Host
   assert(descr.src_ptr != nullptr);
  }

  auto res1 = tn_req->tensor_modes.emplace(std::make_pair(tens_id,std::vector<int32_t>(tens_rank)));
  assert(res1.second);
  for(unsigned int i = 0; i < tens_rank; ++i){
   const auto other_tens_id = tens_legs[i].getTensorId();
   const auto other_tens_leg_id = tens_legs[i].getDimensionId();
   auto other_tens_iter = tn_req->tensor_modes.find(other_tens_id);
   if(other_tens_iter == tn_req->tensor_modes.end()){
    res1.first->second[i] = ++mode_id;
    auto new_mode = tn_req->mode_extents.emplace(std::make_pair(mode_id,tens_dims[i]));
   }else{
    res1.first->second[i] = other_tens_iter->second[other_tens_leg_id];
   }
  }

  if(tens_id == 0){ //output tensor
   tn_req->num_modes_out = tens_rank;
   tn_req->extents_out = res0.first->second.extents.data();
   tn_req->modes_out = res1.first->second.data();
  }else{ //input tensors
   tn_req->num_modes_in[tens_num] = tens_rank;
   tn_req->extents_in[tens_num] = res0.first->second.extents.data();
   tn_req->modes_in[tens_num] = res1.first->second.data();
   ++tens_num;
  }
 }

 const auto tens_elem_type = net.getTensorElementType();
 tn_req->data_type = getCudaDataType(tens_elem_type);
 tn_req->compute_type = getCutensorComputeType(tens_elem_type);

 //Create a cuTensorNet network descriptor for one or all GPUs:
 for(int gpu = 0; gpu < 1; ++gpu){ //`Only one GPU for now
  const auto gpu_id = gpu_attr_[gpu].first;
  HANDLE_CUDA_ERROR(cudaSetDevice(gpu_id));
  HANDLE_CTN_ERROR(cutensornetCreateNetworkDescriptor(gpu_attr_[gpu].second.cutn_handle,num_input_tensors,
                   tn_req->num_modes_in,tn_req->extents_in,tn_req->strides_in,tn_req->modes_in,tn_req->alignments_in,
                   tn_req->num_modes_out,tn_req->extents_out,tn_req->strides_out,tn_req->modes_out,tn_req->alignment_out,
                   tn_req->data_type,tn_req->compute_type,&(tn_req->net_descriptor)));
  HANDLE_CUDA_ERROR(cudaStreamCreate(&(tn_req->stream)));
  HANDLE_CUDA_ERROR(cudaEventCreate(&(tn_req->data_in_start)));
  HANDLE_CUDA_ERROR(cudaEventCreate(&(tn_req->data_in_finish)));
  HANDLE_CUDA_ERROR(cudaEventCreate(&(tn_req->compute_start)));
  HANDLE_CUDA_ERROR(cudaEventCreate(&(tn_req->compute_finish)));
  HANDLE_CUDA_ERROR(cudaEventCreate(&(tn_req->data_out_finish)));
 }
 return;
}


void CuQuantumExecutor::loadTensors(std::shared_ptr<TensorNetworkReq> tn_req)
{
 //Load tensors to one or all GPUs:
 for(int gpu = 0; gpu < 1; ++gpu){ //`Only one GPU for now
  const auto gpu_id = gpu_attr_[gpu].first;
  HANDLE_CUDA_ERROR(cudaSetDevice(gpu_id));
  void * prev_front = mem_pool_[gpu].getFront();
  bool success = true;
  //Acquire device memory:
  for(auto & descr: tn_req->tensor_descriptors){
   void * dev_ptr = mem_pool_[gpu].acquireMemory(descr.second.size);
   success = (dev_ptr != nullptr); if(!success) break;
   descr.second.dst_ptr.emplace_back(dev_ptr);
  }
  if(success){
   //Initiate data transfers:
   HANDLE_CUDA_ERROR(cudaEventRecord(tn_req->data_in_start,tn_req->stream));
   for(auto & descr: tn_req->tensor_descriptors){
    HANDLE_CUDA_ERROR(cudaMemcpyAsync(descr.second.dst_ptr.back(),descr.second.src_ptr,
                                      descr.second.size,cudaMemcpyDefault,tn_req->stream));
   }
   HANDLE_CUDA_ERROR(cudaEventRecord(tn_req->data_in_finish,tn_req->stream));
   tn_req->memory_window_ptr.emplace_back(mem_pool_[gpu].getFront());
   auto & net = *(tn_req->network);
   int32_t tens_num = 0;
   for(auto iter = net.cbegin(); iter != net.cend(); ++iter){
    const auto tens_id = iter->first;
    const auto & tens = iter->second;
    const auto tens_hash = tens.getTensor()->getTensorHash();
    auto descr = tn_req->tensor_descriptors.find(tens_hash);
    void * dev_ptr = descr->second.dst_ptr.back();
    if(tens_id == 0){
     tn_req->data_out = dev_ptr;
    }else{
     tn_req->data_in[tens_num++] = dev_ptr;
    }
   }
  }else{ //no enough memory currently
   //Restore previous memory front:
   mem_pool_[gpu].restorePreviousFront(prev_front);
   return;
  }
 }
 tn_req->exec_status = TensorNetworkQueue::ExecStat::Loading;
 return;
}


void CuQuantumExecutor::planExecution(std::shared_ptr<TensorNetworkReq> tn_req)
{
 //Configure tensor network contraction on one or all GPUs:
 tn_req->prepare_start = Timer::timeInSecHR();
 for(int gpu = 0; gpu < 1; ++gpu){ //`Only one GPU for now
  const auto gpu_id = gpu_attr_[gpu].first;
  HANDLE_CUDA_ERROR(cudaSetDevice(gpu_id));
  HANDLE_CTN_ERROR(cutensornetCreateContractionOptimizerConfig(gpu_attr_[gpu].second.cutn_handle,&(tn_req->opt_config)));
  HANDLE_CTN_ERROR(cutensornetCreateContractionOptimizerInfo(gpu_attr_[gpu].second.cutn_handle,tn_req->net_descriptor,&(tn_req->opt_info)));
  acquireWorkspace(gpu,&(tn_req->workspace),&(tn_req->worksize));
  HANDLE_CTN_ERROR(cutensornetContractionOptimize(gpu_attr_[gpu].second.cutn_handle,
                                                  tn_req->net_descriptor,tn_req->opt_config,
                                                  tn_req->worksize,tn_req->opt_info));
  HANDLE_CTN_ERROR(cutensornetCreateContractionPlan(gpu_attr_[gpu].second.cutn_handle,
                                                    tn_req->net_descriptor,tn_req->opt_info,
                                                    tn_req->worksize,&(tn_req->comp_plan)));
  double flops = 0.0;
  HANDLE_CTN_ERROR(cutensornetContractionOptimizerInfoGetAttribute(gpu_attr_[gpu].second.cutn_handle,
                                                                   tn_req->opt_info,
                                                                   CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT,
                                                                   &flops,sizeof(flops)));
  flops_ += flops;
 }
 tn_req->prepare_finish = Timer::timeInSecHR();
 tn_req->exec_status = TensorNetworkQueue::ExecStat::Planning;
 return;
}


void CuQuantumExecutor::contractTensorNetwork(std::shared_ptr<TensorNetworkReq> tn_req)
{
 //Execute the contraction plan on one or all GPUs:
 for(int gpu = 0; gpu < 1; ++gpu){ //`Only one GPU for now
  const auto gpu_id = gpu_attr_[gpu].first;
  HANDLE_CUDA_ERROR(cudaSetDevice(gpu_id));
  tn_req->num_slices = 0;
  HANDLE_CTN_ERROR(cutensornetContractionOptimizerInfoGetAttribute(gpu_attr_[gpu].second.cutn_handle,
                                                                   tn_req->opt_info,
                                                                   CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES,
                                                                   &(tn_req->num_slices),sizeof(tn_req->num_slices)));
  assert(tn_req->num_slices > 0);
  HANDLE_CUDA_ERROR(cudaEventRecord(tn_req->compute_start,tn_req->stream));
  for(int64_t slice_id = tn_req->proc_id; slice_id < tn_req->num_slices; slice_id += tn_req->num_procs){
   HANDLE_CTN_ERROR(cutensornetContraction(gpu_attr_[gpu].second.cutn_handle,
                                           tn_req->comp_plan,
                                           tn_req->data_in,tn_req->data_out,
                                           tn_req->workspace,tn_req->worksize,
                                           slice_id,tn_req->stream));
  }
  HANDLE_CUDA_ERROR(cudaEventRecord(tn_req->compute_finish,tn_req->stream));
  const auto output_hash = tn_req->network->getTensor(0)->getTensorHash();
  auto iter = tn_req->tensor_descriptors.find(output_hash);
  assert(iter != tn_req->tensor_descriptors.cend());
  const auto & descr = iter->second;
  HANDLE_CUDA_ERROR(cudaMemcpyAsync(descr.src_ptr,descr.dst_ptr[gpu],
                                    descr.size,cudaMemcpyDefault,tn_req->stream));
  HANDLE_CUDA_ERROR(cudaEventRecord(tn_req->data_out_finish,tn_req->stream));
 }
 tn_req->exec_status = TensorNetworkQueue::ExecStat::Executing;
 return;
}


void CuQuantumExecutor::testCompletion(std::shared_ptr<TensorNetworkReq> tn_req)
{
 //Test work completion on one or all GPUs:
 bool all_completed = true;
 for(int gpu = 0; gpu < 1; ++gpu){ //`Only one GPU for now
  const auto gpu_id = gpu_attr_[gpu].first;
  HANDLE_CUDA_ERROR(cudaSetDevice(gpu_id));
  cudaError_t cuda_error = cudaEventQuery(tn_req->data_out_finish);
  if(cuda_error == cudaSuccess){
   if(tn_req->memory_window_ptr[gpu] != nullptr){
    mem_pool_[gpu].releaseMemory(tn_req->memory_window_ptr[gpu]);
    tn_req->memory_window_ptr[gpu] = nullptr;
   }
  }else{
   all_completed = false;
  }
 }
 if(all_completed) tn_req->exec_status = TensorNetworkQueue::ExecStat::Completed;
 return;
}

} //namespace runtime
} //namespace exatn

#endif //CUQUANTUM
