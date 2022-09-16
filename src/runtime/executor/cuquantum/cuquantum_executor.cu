/** ExaTN: Tensor Runtime: Tensor network executor: NVIDIA cuQuantum
REVISION: 2022/09/16

Copyright (C) 2018-2022 Dmitry Lyakh
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle)
Copyright (C) 2022-2022 NVIDIA Corporation

SPDX-License-Identifier: BSD-3-Clause **/

#ifdef CUQUANTUM

#include <cutensornet.h>
#include <cutensor.h>
#include <cuda_runtime.h>

#ifdef MPI_ENABLED
#include "mpi.h"
#endif

#include <algorithm>
#include <vector>
#include <unordered_map>
#include <numeric>
#include <type_traits>
#include <cstdint>
#include <complex>
#include <iostream>

#include "byte_packet.h"
#include "talshxx.hpp"

#include "timers.hpp"

#include "cuquantum_executor.hpp"


#define HANDLE_CUDA_ERROR(x) \
{ const auto err = x; \
  if( err != cudaSuccess ) \
  { printf("#ERROR(cuquantum_executor): %s in line %d\n", cudaGetErrorString(err), __LINE__); fflush(stdout); std::abort(); } \
};

#define HANDLE_CTN_ERROR(x) \
{ const auto err = x; \
  if( err != CUTENSORNET_STATUS_SUCCESS ) \
  { printf("#ERROR(cuquantum_executor): %s in line %d\n", cutensornetGetErrorString(err), __LINE__); fflush(stdout); std::abort(); } \
};


namespace exatn {
namespace runtime {

/** Retrieves a state of cutensornetContractionOptimizerInfo_t as a plain byte packet. **/
void getCutensornetContractionOptimizerInfoState(cutensornetHandle_t & handle,                 //cuTensorNet handle
                                                 cutensornetContractionOptimizerInfo_t & info, //in: cutensornetContractionOptimizerInfo_t object
                                                 BytePacket * info_state);                     //out: state of the object as a plain byte packet

/** Sets a state of cutensornetContractionOptimizerInfo_t from a plain byte packet. **/
void setCutensornetContractionOptimizerInfoState(cutensornetHandle_t & handle,                 //cuTensorNet handle
                                                 cutensornetContractionOptimizerInfo_t & info, //out: cutensornetContractionOptimizerInfo_t object
                                                 BytePacket * info_state);                     //in: state of the object as a plain byte packet
#ifdef MPI_ENABLED
/** Broadcasts a cutensornetContractionOptimizerInfo_t to all MPI processes. **/
void broadcastCutensornetContractionOptimizerInfo(cutensornetHandle_t & handle,                 //cuTensorNet handle
                                                  cutensornetContractionOptimizerInfo_t & info, //in: cutensornetContractionOptimizerInfo_t object
                                                  MPICommProxy & communicator);                 //in: MPI communicator
#endif


/** Tensor descriptor (inside a tensor network) **/
struct TensorDescriptor {
 std::vector<int64_t> extents; //tensor dimension extents
 std::vector<int64_t> strides; //tensor dimension strides (optional)
 cudaDataType_t data_type;     //tensor element data type
 std::size_t volume = 0;       //tensor body volume
 std::size_t size = 0;         //tensor body size (bytes)
 void * src_ptr = nullptr;     //non-owning pointer to the tensor body source image
 std::vector<void*> dst_ptr;   //non-owning pointer to the tensor body destination image on each GPU
};

/** Tensor network processing request **/
struct TensorNetworkReq {
 TensorNetworkQueue::ExecStat exec_status = TensorNetworkQueue::ExecStat::None; //tensor network execution status
 int num_procs = 0; //total number of executing processes
 int proc_id = -1; //id of the current executing process
#ifdef MPI_ENABLED
 MPICommProxy comm; //MPI communicator over executing processes
#endif
 int64_t num_slices = 0;
 std::shared_ptr<numerics::TensorNetwork> network; //original tensor network specification
 std::unordered_map<numerics::TensorHashType, TensorDescriptor> tensor_descriptors; //tensor descriptors (shape, volume, data type, body)
 std::unordered_map<unsigned int, std::vector<int32_t>> tensor_modes; //indices associated with tensor dimensions (key = original tensor id)
 std::unordered_map<int32_t, int64_t> mode_extents; //extent of each registered tensor mode (mode --> extent)
 int32_t * num_modes_in = nullptr;
 int64_t ** extents_in = nullptr;
 int64_t ** strides_in = nullptr;
 int32_t ** modes_in = nullptr;
 uint32_t * alignments_in = nullptr;
 std::vector<void**> gpu_data_in; //vector of owning arrays of non-owning pointers to the input tensor bodies on each GPU
 int32_t num_modes_out;
 int64_t * extents_out = nullptr; //non-owning
 int64_t * strides_out = nullptr;
 int32_t * modes_out = nullptr; //non-owning
 uint32_t alignment_out;
 std::vector<void*> gpu_data_out; //vector of non-owning pointers to the output tensor body on each GPU
 std::vector<void*> gpu_workspace; //vector of non-owning pointers to the work space on each GPU
 std::vector<uint64_t> gpu_worksize; //work space size on each GPU
 std::vector<void*> memory_window_ptr; //end of the GPU memory segment allocated for the tensors on each GPU
 cutensornetNetworkDescriptor_t net_descriptor;
 cutensornetContractionOptimizerConfig_t opt_config;
 cutensornetContractionOptimizerInfo_t opt_info;
 std::vector<cutensornetWorkspaceDescriptor_t> workspace_descriptor; //for each GPU
 std::vector<cutensornetContractionPlan_t> comp_plan; //for each GPU
 cudaDataType_t data_type;
 cutensornetComputeType_t compute_type;
 std::vector<cudaStream_t> gpu_stream; //CUDA stream on each GPU
 std::vector<cudaEvent_t> gpu_data_in_start; //event on each GPU
 std::vector<cudaEvent_t> gpu_data_in_finish; //event on each GPU
 std::vector<cudaEvent_t> gpu_compute_start; //event on each GPU
 std::vector<cudaEvent_t> gpu_compute_finish; //event on each GPU
 std::vector<cudaEvent_t> gpu_data_out_finish; //event on each GPU
 double prepare_start;
 double prepare_finish;

 ~TensorNetworkReq() {
  for(auto & stream: gpu_stream) cudaStreamSynchronize(stream);
  for(auto & event: gpu_data_out_finish) cudaEventDestroy(event);
  for(auto & event: gpu_compute_finish) cudaEventDestroy(event);
  for(auto & event: gpu_compute_start) cudaEventDestroy(event);
  for(auto & event: gpu_data_in_finish) cudaEventDestroy(event);
  for(auto & event: gpu_data_in_start) cudaEventDestroy(event);
  for(auto & stream: gpu_stream) cudaStreamDestroy(stream);
  for(auto & plan: comp_plan) cutensornetDestroyContractionPlan(plan);
  for(auto & ws_descr: workspace_descriptor) cutensornetDestroyWorkspaceDescriptor(ws_descr);
  cutensornetDestroyContractionOptimizerInfo(opt_info);
  cutensornetDestroyContractionOptimizerConfig(opt_config);
  cutensornetDestroyNetworkDescriptor(net_descriptor);
  //if(modes_out != nullptr) delete [] modes_out;
  if(strides_out != nullptr) delete [] strides_out;
  //if(extents_out != nullptr) delete [] extents_out;
  for(auto & data_in: gpu_data_in) if(data_in != nullptr) delete [] data_in;
  if(alignments_in != nullptr) delete [] alignments_in;
  if(modes_in != nullptr) delete [] modes_in;
  if(strides_in != nullptr) delete [] strides_in;
  if(extents_in != nullptr) delete [] extents_in;
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
 if(process_rank_ == 0){
  std::cout << "#DEBUG(exatn::runtime::CuQuantumExecutor): cuTensorNet backend version " << version << std::endl;
  std::cout << "#DEBUG(exatn::runtime::CuQuantumExecutor): Total number of processes = " << num_processes_ << std::endl;
 }

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
 if(process_rank_ == 0)
  std::cout << "#DEBUG(exatn::runtime::CuQuantumExecutor): Number of available GPUs = " << gpu_attr_.size() << std::endl;
 if(gpu_attr_.empty()){
  fatal_error("#FATAL(exatn::runtime::CuQuantumExecutor): cuQuantum backend requires at least one NVIDIA GPU per MPI process!\n");
 }

 for(const auto & gpu: gpu_attr_){
  HANDLE_CUDA_ERROR(cudaSetDevice(gpu.first));
  HANDLE_CTN_ERROR(cutensornetCreate((cutensornetHandle_t*)(&gpu.second.cutn_handle)));
 }
 if(process_rank_ == 0){
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
}


CuQuantumExecutor::~CuQuantumExecutor()
{
 sync();
 for(const auto & gpu: gpu_attr_){
  HANDLE_CUDA_ERROR(cudaSetDevice(gpu.first));
  HANDLE_CTN_ERROR(cutensornetDestroy((cutensornetHandle_t)(gpu.second.cutn_handle)));
 }
 if(process_rank_ == 0)
  std::cout << "#DEBUG(exatn::runtime::CuQuantumExecutor): Destroyed cuTensorNet contexts for all available GPUs" << std::endl;

 std::cout << "#MSG(exatn::cuQuantum): Statistics across all GPU devices:\n";
 std::cout << " Number of Flops processed: " << flops_ << std::endl;
 std::cout << "#END_MSG\n";
 gpu_attr_.clear();
}


#ifdef MPI_ENABLED
TensorNetworkQueue::ExecStat CuQuantumExecutor::execute(std::shared_ptr<numerics::TensorNetwork> network,
                                                        unsigned int num_processes, unsigned int process_rank,
                                                        const MPICommProxy & communicator,
                                                        const TensorOpExecHandle exec_handle)
#else
TensorNetworkQueue::ExecStat CuQuantumExecutor::execute(std::shared_ptr<numerics::TensorNetwork> network,
                                                        unsigned int num_processes, unsigned int process_rank,
                                                        const TensorOpExecHandle exec_handle)
#endif
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
#ifdef MPI_ENABLED
  tn_req->comm = communicator;
#endif
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
                                                     std::vector<ExecutionTimings> * timings)
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
    const int num_gpus = gpu_attr_.size();
    (*timings).resize(num_gpus);
    for(int gpu = 0; gpu < num_gpus; ++gpu){
     (*timings)[gpu].prepare = (tn_req->prepare_finish - tn_req->prepare_start) * 1000.0; //ms
     HANDLE_CUDA_ERROR(cudaEventElapsedTime(&((*timings)[gpu].data_in),
                       tn_req->gpu_data_in_start[gpu],tn_req->gpu_data_in_finish[gpu]));
     HANDLE_CUDA_ERROR(cudaEventElapsedTime(&((*timings)[gpu].data_out),
                       tn_req->gpu_compute_finish[gpu],tn_req->gpu_data_out_finish[gpu]));
     HANDLE_CUDA_ERROR(cudaEventElapsedTime(&((*timings)[gpu].compute),
                       tn_req->gpu_compute_start[gpu],tn_req->gpu_compute_finish[gpu]));
    }
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


static cudaDataType_t getCudaDataType(const TensorElementType elem_type)
{
 cudaDataType_t cuda_data_type;
 switch(elem_type){
  case TensorElementType::REAL32: cuda_data_type = CUDA_R_32F; break;
  case TensorElementType::REAL64: cuda_data_type = CUDA_R_64F; break;
  case TensorElementType::COMPLEX32: cuda_data_type = CUDA_C_32F; break;
  case TensorElementType::COMPLEX64: cuda_data_type = CUDA_C_64F; break;
  default: assert(false);
 }
 return cuda_data_type;
}


static cutensornetComputeType_t getCutensorComputeType(const TensorElementType elem_type)
{
 cutensornetComputeType_t cutensor_data_type;
 switch(elem_type){
  case TensorElementType::REAL32: cutensor_data_type = CUTENSORNET_COMPUTE_32F; break;
  case TensorElementType::REAL64: cutensor_data_type = CUTENSORNET_COMPUTE_64F; break;
  case TensorElementType::COMPLEX32: cutensor_data_type = CUTENSORNET_COMPUTE_32F; break;
  case TensorElementType::COMPLEX64: cutensor_data_type = CUTENSORNET_COMPUTE_64F; break;
  default: assert(false);
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
 const int num_gpus = gpu_attr_.size();
 const auto & net = *(tn_req->network);
 const int32_t num_input_tensors = net.getNumTensors();

 tn_req->num_modes_in = new int32_t[num_input_tensors];
 tn_req->extents_in = new int64_t*[num_input_tensors];
 tn_req->strides_in = new int64_t*[num_input_tensors];
 tn_req->modes_in = new int32_t*[num_input_tensors];
 tn_req->alignments_in = new uint32_t[num_input_tensors];

 tn_req->gpu_data_in.resize(num_gpus,nullptr);
 for(auto & data_in: tn_req->gpu_data_in) data_in = new void*[num_input_tensors];
 tn_req->gpu_data_out.resize(num_gpus,nullptr);
 tn_req->gpu_workspace.resize(num_gpus,nullptr);
 tn_req->gpu_worksize.resize(num_gpus,0);

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
   std::abort();
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

 //Create the GPU execution plan, stream and events on each GPU:
 tn_req->workspace_descriptor.resize(num_gpus);
 tn_req->comp_plan.resize(num_gpus);
 tn_req->gpu_stream.resize(num_gpus);
 tn_req->gpu_data_in_start.resize(num_gpus);
 tn_req->gpu_data_in_finish.resize(num_gpus);
 tn_req->gpu_compute_start.resize(num_gpus);
 tn_req->gpu_compute_finish.resize(num_gpus);
 tn_req->gpu_data_out_finish.resize(num_gpus);
 for(int gpu = 0; gpu < num_gpus; ++gpu){
  const auto gpu_id = gpu_attr_[gpu].first;
  HANDLE_CUDA_ERROR(cudaSetDevice(gpu_id));
  HANDLE_CUDA_ERROR(cudaStreamCreate(&(tn_req->gpu_stream[gpu])));
  HANDLE_CUDA_ERROR(cudaEventCreate(&(tn_req->gpu_data_in_start[gpu])));
  HANDLE_CUDA_ERROR(cudaEventCreate(&(tn_req->gpu_data_in_finish[gpu])));
  HANDLE_CUDA_ERROR(cudaEventCreate(&(tn_req->gpu_compute_start[gpu])));
  HANDLE_CUDA_ERROR(cudaEventCreate(&(tn_req->gpu_compute_finish[gpu])));
  HANDLE_CUDA_ERROR(cudaEventCreate(&(tn_req->gpu_data_out_finish[gpu])));
 }

 //Create the cuTensorNet tensor network descriptor (not GPU specific):
 HANDLE_CTN_ERROR(cutensornetCreateNetworkDescriptor(gpu_attr_[0].second.cutn_handle,num_input_tensors,
                  tn_req->num_modes_in,tn_req->extents_in,tn_req->strides_in,tn_req->modes_in,tn_req->alignments_in,
                  tn_req->num_modes_out,tn_req->extents_out,tn_req->strides_out,tn_req->modes_out,tn_req->alignment_out,
                  tn_req->data_type,tn_req->compute_type,&(tn_req->net_descriptor)));
 return;
}


void CuQuantumExecutor::loadTensors(std::shared_ptr<TensorNetworkReq> tn_req)
{
 const auto out_tens_hash = tn_req->network->getTensor(0)->getTensorHash();
 //Load tensors to all GPUs:
 const int num_gpus = gpu_attr_.size();
 for(int gpu = 0; gpu < num_gpus; ++gpu){
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
   HANDLE_CUDA_ERROR(cudaEventRecord(tn_req->gpu_data_in_start[gpu],tn_req->gpu_stream[gpu]));
   for(auto & descr: tn_req->tensor_descriptors){
    if(descr.first == out_tens_hash){ //output tensor: Set to 0
     HANDLE_CUDA_ERROR(cudaMemsetAsync(descr.second.dst_ptr.back(),0,descr.second.size,tn_req->gpu_stream[gpu]));
    }else{ //input tensors: Copy from their original locations
     /*std::cout << "#DEBUG(exatn::CuQuantumExecutor): loadTensors: "
                 << descr.second.dst_ptr.back() << " " << descr.second.src_ptr << " "
                 << descr.second.size << std::endl << std::flush; //debug*/
     HANDLE_CUDA_ERROR(cudaMemcpyAsync(descr.second.dst_ptr.back(),descr.second.src_ptr,
                                       descr.second.size,cudaMemcpyDefault,tn_req->gpu_stream[gpu]));
    }
   }
   HANDLE_CUDA_ERROR(cudaEventRecord(tn_req->gpu_data_in_finish[gpu],tn_req->gpu_stream[gpu]));
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
     tn_req->gpu_data_out[gpu] = dev_ptr;
    }else{
     tn_req->gpu_data_in[gpu][tens_num++] = dev_ptr;
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


void getCutensornetContractionOptimizerInfoState(cutensornetHandle_t & handle,
                                                 cutensornetContractionOptimizerInfo_t & info,
                                                 BytePacket * info_state)
{
 cutensornetContractionPath_t contr_path{0,nullptr};
 HANDLE_CTN_ERROR(cutensornetContractionOptimizerInfoGetAttribute(handle,
                                                                  info,
                                                                  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH,
                                                                  &contr_path,sizeof(contr_path)));
 assert(contr_path.numContractions >= 0);
 appendToBytePacket(info_state,contr_path.numContractions);
 if(contr_path.numContractions > 0){
  contr_path.data = new cutensornetNodePair_t[contr_path.numContractions];
  assert(contr_path.data != nullptr);
  HANDLE_CTN_ERROR(cutensornetContractionOptimizerInfoGetAttribute(handle,
                                                                   info,
                                                                   CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH,
                                                                   &contr_path,sizeof(contr_path)));
  for(int32_t i = 0; i < contr_path.numContractions; ++i){
   appendToBytePacket(info_state,contr_path.data[i].first);
   appendToBytePacket(info_state,contr_path.data[i].second);
  }
  int32_t num_sliced_modes = 0;
  HANDLE_CTN_ERROR(cutensornetContractionOptimizerInfoGetAttribute(handle,
                                                                   info,
                                                                   CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICED_MODES,
                                                                   &num_sliced_modes,sizeof(num_sliced_modes)));
  assert(num_sliced_modes >= 0);
  appendToBytePacket(info_state,num_sliced_modes);
  if(num_sliced_modes > 0){
   std::vector<int32_t> sliced_modes(num_sliced_modes);
   HANDLE_CTN_ERROR(cutensornetContractionOptimizerInfoGetAttribute(handle,
                                                                    info,
                                                                    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_MODE,
                                                                    sliced_modes.data(),sliced_modes.size()*sizeof(int32_t)));
   for(int32_t i = 0; i < num_sliced_modes; ++i){
    appendToBytePacket(info_state,sliced_modes[i]);
   }
   std::vector<int64_t> sliced_extents(num_sliced_modes);
   HANDLE_CTN_ERROR(cutensornetContractionOptimizerInfoGetAttribute(handle,
                                                                    info,
                                                                    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_EXTENT,
                                                                    sliced_extents.data(),sliced_extents.size()*sizeof(int64_t)));
   for(int32_t i = 0; i < num_sliced_modes; ++i){
    appendToBytePacket(info_state,sliced_extents[i]);
   }
  }
  if(contr_path.data != nullptr) delete [] contr_path.data;
 }
 return;
}


void setCutensornetContractionOptimizerInfoState(cutensornetHandle_t & handle,
                                                 cutensornetContractionOptimizerInfo_t & info,
                                                 BytePacket * info_state)
{
 cutensornetContractionPath_t contr_path{0,nullptr};
 HANDLE_CTN_ERROR(cutensornetContractionOptimizerInfoGetAttribute(handle,
                                                                  info,
                                                                  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH,
                                                                  &contr_path,sizeof(contr_path)));
 assert(contr_path.numContractions >= 0);
 int32_t num_contractions = 0;
 extractFromBytePacket(info_state,num_contractions);
 assert(num_contractions == contr_path.numContractions);
 if(contr_path.numContractions > 0){
  contr_path.data = new cutensornetNodePair_t[contr_path.numContractions];
  assert(contr_path.data != nullptr);
  int32_t first, second;
  for(int32_t i = 0; i < contr_path.numContractions; ++i){
   extractFromBytePacket(info_state,first);
   extractFromBytePacket(info_state,second);
   contr_path.data[i].first = first;
   contr_path.data[i].second = second;
  }
  HANDLE_CTN_ERROR(cutensornetContractionOptimizerInfoSetAttribute(handle,
                                                                   info,
                                                                   CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH,
                                                                   &contr_path,sizeof(contr_path)));
  int32_t num_sliced_modes = 0;
  extractFromBytePacket(info_state,num_sliced_modes);
  assert(num_sliced_modes >= 0);
  HANDLE_CTN_ERROR(cutensornetContractionOptimizerInfoSetAttribute(handle,
                                                                   info,
                                                                   CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICED_MODES,
                                                                   &num_sliced_modes,sizeof(num_sliced_modes)));
  if(num_sliced_modes > 0){
   std::vector<int32_t> sliced_modes(num_sliced_modes);
   for(int32_t i = 0; i < num_sliced_modes; ++i){
    extractFromBytePacket(info_state,sliced_modes[i]);
   }
   HANDLE_CTN_ERROR(cutensornetContractionOptimizerInfoSetAttribute(handle,
                                                                    info,
                                                                    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_MODE,
                                                                    sliced_modes.data(),sliced_modes.size()*sizeof(int32_t)));
   std::vector<int64_t> sliced_extents(num_sliced_modes);
   for(int32_t i = 0; i < num_sliced_modes; ++i){
    extractFromBytePacket(info_state,sliced_extents[i]);
   }
   HANDLE_CTN_ERROR(cutensornetContractionOptimizerInfoSetAttribute(handle,
                                                                    info,
                                                                    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_EXTENT,
                                                                    sliced_extents.data(),sliced_extents.size()*sizeof(int64_t)));
  }
  if(contr_path.data != nullptr) delete [] contr_path.data;
 }
 return;
}


#ifdef MPI_ENABLED
void broadcastCutensornetContractionOptimizerInfo(cutensornetHandle_t & handle,
                                                  cutensornetContractionOptimizerInfo_t & info,
                                                  MPICommProxy & communicator)
{
 double flops = 0.0;
 HANDLE_CTN_ERROR(cutensornetContractionOptimizerInfoGetAttribute(handle,
                                                                  info,
                                                                  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT,
                                                                  &flops,sizeof(flops)));
 assert(flops >= 0.0);
 auto & mpi_comm = communicator.getRef<MPI_Comm>();
 int my_rank = -1;
 auto errc = MPI_Comm_rank(mpi_comm, &my_rank);
 assert(errc == MPI_SUCCESS);
 struct {double flop_count; int mpi_rank;} my_flop{flops,my_rank}, best_flop{0.0,-1};
 errc = MPI_Allreduce((void*)(&my_flop),(void*)(&best_flop),1,MPI_DOUBLE_INT,MPI_MINLOC,mpi_comm);
 assert(errc == MPI_SUCCESS);

 cutensornetContractionPath_t contr_path{0,nullptr};
 HANDLE_CTN_ERROR(cutensornetContractionOptimizerInfoGetAttribute(handle,
                                                                  info,
                                                                  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH,
                                                                  &contr_path,sizeof(contr_path)));
 assert(contr_path.numContractions >= 0);
 if(contr_path.numContractions > 0){
  contr_path.data = new cutensornetNodePair_t[contr_path.numContractions];
  assert(contr_path.data != nullptr);
  HANDLE_CTN_ERROR(cutensornetContractionOptimizerInfoGetAttribute(handle,
                                                                   info,
                                                                   CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH,
                                                                   &contr_path,sizeof(contr_path)));
  int32_t num_sliced_modes = 0;
  HANDLE_CTN_ERROR(cutensornetContractionOptimizerInfoGetAttribute(handle,
                                                                   info,
                                                                   CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICED_MODES,
                                                                   &num_sliced_modes,sizeof(num_sliced_modes)));
  std::size_t packet_capacity = ((sizeof(int32_t) * 2 * contr_path.numContractions) +
                                 ((sizeof(int32_t) + sizeof(int64_t)) * num_sliced_modes)) * 2 + 1024; //upper bound
  BytePacket packet;
  initBytePacket(&packet,packet_capacity);
  if(my_rank == best_flop.mpi_rank){
   getCutensornetContractionOptimizerInfoState(handle,info,&packet);
  }
  int packet_size = packet.size_bytes;
  errc = MPI_Bcast((void*)(&packet_size),1,MPI_INT,best_flop.mpi_rank,mpi_comm);
  assert(errc == MPI_SUCCESS);
  if(my_rank != best_flop.mpi_rank){
   packet.size_bytes = packet_size;
  }
  errc = MPI_Bcast(packet.base_addr,packet_size,MPI_CHAR,best_flop.mpi_rank,mpi_comm);
  assert(errc == MPI_SUCCESS);
  if(my_rank != best_flop.mpi_rank){
   setCutensornetContractionOptimizerInfoState(handle,info,&packet);
  }
  destroyBytePacket(&packet);
  if(contr_path.data != nullptr) delete [] contr_path.data;
 }
 return;
}
#endif


void CuQuantumExecutor::planExecution(std::shared_ptr<TensorNetworkReq> tn_req)
{
 //Configure tensor network contraction on all GPUs:
 tn_req->prepare_start = Timer::timeInSecHR();
 const int num_gpus = gpu_attr_.size();
 for(int gpu = 0; gpu < num_gpus; ++gpu){
  const int gpu_id = gpu_attr_[gpu].first;
  HANDLE_CUDA_ERROR(cudaSetDevice(gpu_id));
  acquireWorkspace(gpu,&(tn_req->gpu_workspace[gpu]),&(tn_req->gpu_worksize[gpu]));
 }
 const auto min_gpu_workspace_size = *(std::min_element(tn_req->gpu_worksize.cbegin(),tn_req->gpu_worksize.cend()));
 for(int gpu = 0; gpu < num_gpus; ++gpu){
  const int gpu_id = gpu_attr_[gpu].first;
  HANDLE_CUDA_ERROR(cudaSetDevice(gpu_id));
  if(gpu == 0){ //tensor network contraction path needs to be computed only once
   const int32_t min_slices = tn_req->num_procs * num_gpus; //ensure parallelism
   HANDLE_CTN_ERROR(cutensornetCreateContractionOptimizerConfig(gpu_attr_[gpu].second.cutn_handle,&(tn_req->opt_config)));
   HANDLE_CTN_ERROR(cutensornetContractionOptimizerConfigSetAttribute(gpu_attr_[gpu].second.cutn_handle,tn_req->opt_config,
                                                                      CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MIN_SLICES,
                                                                      &min_slices,sizeof(min_slices)));
   const cutensornetOptimizerCost_t cost_func = CUTENSORNET_OPTIMIZER_COST_TIME;
   HANDLE_CTN_ERROR(cutensornetContractionOptimizerConfigSetAttribute(gpu_attr_[gpu].second.cutn_handle,tn_req->opt_config,
                                                                      CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_COST_FUNCTION_OBJECTIVE,
                                                                      &cost_func,sizeof(cost_func)));
   const int32_t hyper_samples = 8;
   HANDLE_CTN_ERROR(cutensornetContractionOptimizerConfigSetAttribute(gpu_attr_[gpu].second.cutn_handle,tn_req->opt_config,
                                                                      CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES,
                                                                      &hyper_samples,sizeof(hyper_samples)));
   const int32_t reconfig_iter = 256;
   HANDLE_CTN_ERROR(cutensornetContractionOptimizerConfigSetAttribute(gpu_attr_[gpu].second.cutn_handle,tn_req->opt_config,
                                                                      CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_ITERATIONS,
                                                                      &reconfig_iter,sizeof(reconfig_iter)));
   HANDLE_CTN_ERROR(cutensornetCreateContractionOptimizerInfo(gpu_attr_[gpu].second.cutn_handle,tn_req->net_descriptor,&(tn_req->opt_info)));
   HANDLE_CTN_ERROR(cutensornetContractionOptimize(gpu_attr_[gpu].second.cutn_handle,
                                                   tn_req->net_descriptor,tn_req->opt_config,
                                                   min_gpu_workspace_size,tn_req->opt_info));
#ifdef MPI_ENABLED
   if(tn_req->num_procs > 1){
    broadcastCutensornetContractionOptimizerInfo(gpu_attr_[gpu].second.cutn_handle,tn_req->opt_info,tn_req->comm);
   }
#endif
   double flops = 0.0;
   HANDLE_CTN_ERROR(cutensornetContractionOptimizerInfoGetAttribute(gpu_attr_[gpu].second.cutn_handle,
                                                                    tn_req->opt_info,
                                                                    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT,
                                                                    &flops,sizeof(flops)));
   assert(flops >= 0.0);
   flops_ += ((flops * 0.5) / static_cast<double>(tn_req->num_procs)) * //assuming uniform work distribution
             tensorElementTypeOpFactor(tn_req->network->getTensorElementType());
   tn_req->num_slices = 0;
   HANDLE_CTN_ERROR(cutensornetContractionOptimizerInfoGetAttribute(gpu_attr_[gpu].second.cutn_handle,
                                                                    tn_req->opt_info,
                                                                    CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES,
                                                                    &(tn_req->num_slices),sizeof(tn_req->num_slices)));
   assert(tn_req->num_slices > 0);
  }
  HANDLE_CTN_ERROR(cutensornetCreateWorkspaceDescriptor(gpu_attr_[gpu].second.cutn_handle,&(tn_req->workspace_descriptor[gpu])));
  HANDLE_CTN_ERROR(cutensornetWorkspaceComputeSizes(gpu_attr_[gpu].second.cutn_handle,
                                                    tn_req->net_descriptor,tn_req->opt_info,
                                                    tn_req->workspace_descriptor[gpu]));
  uint64_t required_workspace_size = 0;
  HANDLE_CTN_ERROR(cutensornetWorkspaceGetSize(gpu_attr_[gpu].second.cutn_handle,
                                               tn_req->workspace_descriptor[gpu],
                                               CUTENSORNET_WORKSIZE_PREF_MIN,
                                               CUTENSORNET_MEMSPACE_DEVICE,
                                               &required_workspace_size));
  if(required_workspace_size > tn_req->gpu_worksize[gpu]){
   fatal_error("#ERROR(exatn::CuQuantumExecutor::planExecution): Insufficient work space on GPU "+std::to_string(gpu)+"!\n");
  }
  HANDLE_CTN_ERROR(cutensornetWorkspaceSet(gpu_attr_[gpu].second.cutn_handle,
                                           tn_req->workspace_descriptor[gpu],
                                           CUTENSORNET_MEMSPACE_DEVICE,
                                           tn_req->gpu_workspace[gpu],tn_req->gpu_worksize[gpu]));
  HANDLE_CTN_ERROR(cutensornetCreateContractionPlan(gpu_attr_[gpu].second.cutn_handle,
                                                    tn_req->net_descriptor,tn_req->opt_info,
                                                    tn_req->workspace_descriptor[gpu],&(tn_req->comp_plan[gpu])));
 }
 tn_req->prepare_finish = Timer::timeInSecHR();
 tn_req->exec_status = TensorNetworkQueue::ExecStat::Planning;
 return;
}


void accumulateOutputOnHost(TensorElementType elem_type, void * out_ptr, void * tmp_ptr, std::size_t vol)
{
 auto accumulate = [](auto * ptr0, const auto * ptr1, auto count){
#pragma omp parallel for schedule(guided) shared(count,ptr0,ptr1)
  for(std::size_t i = 0; i < count; ++i) ptr0[i] += ptr1[i];
  return;
 };

 switch(elem_type){
  case TensorElementType::REAL32:
   accumulate(static_cast<float*>(out_ptr),static_cast<float*>(tmp_ptr),vol);
   break;
  case TensorElementType::REAL64:
   accumulate(static_cast<double*>(out_ptr),static_cast<double*>(tmp_ptr),vol);
   break;
  case TensorElementType::COMPLEX32:
   accumulate(static_cast<std::complex<float>*>(out_ptr),static_cast<std::complex<float>*>(tmp_ptr),vol);
   break;
  case TensorElementType::COMPLEX64:
   accumulate(static_cast<std::complex<double>*>(out_ptr),static_cast<std::complex<double>*>(tmp_ptr),vol);
   break;
  default: assert(false);
 }
 return;
}


void CuQuantumExecutor::contractTensorNetwork(std::shared_ptr<TensorNetworkReq> tn_req)
{
 //Execute the contraction plans on all GPUs:
 const int num_gpus = gpu_attr_.size();
 const int64_t total_gpus = tn_req->num_procs * num_gpus;
 for(int gpu = 0; gpu < num_gpus; ++gpu){
  HANDLE_CUDA_ERROR(cudaEventRecord(tn_req->gpu_compute_start[gpu],tn_req->gpu_stream[gpu]));
 }
 for(int64_t slice_base_id = tn_req->proc_id * num_gpus; slice_base_id < tn_req->num_slices; slice_base_id += total_gpus){
  const int64_t slice_end = std::min(tn_req->num_slices,(int64_t)(slice_base_id + num_gpus));
  for(int64_t slice_id = slice_base_id; slice_id < slice_end; ++slice_id){
   const int gpu = static_cast<int>(slice_id - slice_base_id);
   HANDLE_CTN_ERROR(cutensornetContraction(gpu_attr_[gpu].second.cutn_handle,
                                           tn_req->comp_plan[gpu],
                                           tn_req->gpu_data_in[gpu],tn_req->gpu_data_out[gpu],
                                           tn_req->workspace_descriptor[gpu],
                                           slice_id,tn_req->gpu_stream[gpu]));
  }
 }
 for(int gpu = 0; gpu < num_gpus; ++gpu){
  HANDLE_CUDA_ERROR(cudaEventRecord(tn_req->gpu_compute_finish[gpu],tn_req->gpu_stream[gpu]));
 }
 //Retrieve the output tensor from all GPUs and perform reduction:
 auto output_tensor = tn_req->network->getTensor(0);
 const auto out_elem_type = output_tensor->getElementType();
 const auto out_elem_size = TensorElementTypeSize(out_elem_type);
 const auto output_hash = output_tensor->getTensorHash();
 auto iter = tn_req->tensor_descriptors.find(output_hash);
 assert(iter != tn_req->tensor_descriptors.cend());
 const auto & descr = iter->second;
 if(num_gpus > 1){ //`Blocking solution for output reduction (temporary)
  const auto dev_id = talshFlatDevId(DEV_HOST,0);
  void * host_out_tens = nullptr;
  auto errc = mem_allocate(dev_id,descr.size,YEP,&host_out_tens);
  if(errc != 0 || host_out_tens == nullptr){
   fatal_error("#ERROR(exatn::CuQuantumExecutor::contractTensorNetwork): Insufficient memory space in the Host buffer!\n");
  }
  struct ReductionBuf{void * tmp_ptr; void * out_ptr; void * gpu_ptr; std::size_t vol;};
  assert(MEM_ALIGNMENT % out_elem_size == 0);
  const auto vol0 = (descr.volume / 2) - ((descr.volume / 2) % (MEM_ALIGNMENT / out_elem_size));
  const auto vol1 = descr.volume - vol0;
  std::vector<ReductionBuf> red_buf = {ReductionBuf{host_out_tens,
                                                    descr.src_ptr,
                                                    nullptr,
                                                    vol0},
                                       ReductionBuf{(void*)(((char*)host_out_tens)+(vol0*out_elem_size)),
                                                    (void*)(((char*)descr.src_ptr)+(vol0*out_elem_size)),
                                                    nullptr,
                                                    vol1}};
  bool first_iteration = true;
  for(int gpu = 0; gpu < num_gpus; ++gpu){
   red_buf[0].gpu_ptr = descr.dst_ptr[gpu];
   red_buf[1].gpu_ptr = (void*)(((char*)descr.dst_ptr[gpu])+(vol0*out_elem_size));
   for(int part = 0; part < 2; ++part){
    HANDLE_CUDA_ERROR(cudaMemcpyAsync(red_buf[part].tmp_ptr,red_buf[part].gpu_ptr,
                                      red_buf[part].vol*out_elem_size,cudaMemcpyDefault,tn_req->gpu_stream[gpu]));
    if(first_iteration){
     HANDLE_CUDA_ERROR(cudaEventRecord(tn_req->gpu_data_out_finish[gpu],tn_req->gpu_stream[gpu]));
     first_iteration = false;
    }else{
     if(part == 0){
      HANDLE_CUDA_ERROR(cudaEventSynchronize(tn_req->gpu_data_out_finish[gpu-1]));
     }else{
      HANDLE_CUDA_ERROR(cudaEventSynchronize(tn_req->gpu_data_out_finish[gpu]));
     }
     HANDLE_CUDA_ERROR(cudaEventRecord(tn_req->gpu_data_out_finish[gpu],tn_req->gpu_stream[gpu]));
     const auto other_part = 1 - part;
     accumulateOutputOnHost(out_elem_type,red_buf[other_part].out_ptr,red_buf[other_part].tmp_ptr,red_buf[other_part].vol);
    }
   }
  }
  errc = mem_free(dev_id,&host_out_tens);
  if(errc != 0){
   fatal_error("#ERROR(exatn::CuQuantumExecutor::contractTensorNetwork): Unable to free a temporary Host buffer entry!\n");
  }
 }else{
  HANDLE_CUDA_ERROR(cudaMemcpyAsync(descr.src_ptr,descr.dst_ptr[0],
                                    descr.size,cudaMemcpyDefault,tn_req->gpu_stream[0]));
  HANDLE_CUDA_ERROR(cudaEventRecord(tn_req->gpu_data_out_finish[0],tn_req->gpu_stream[0]));
 }
 tn_req->exec_status = TensorNetworkQueue::ExecStat::Executing;
 return;
}


void CuQuantumExecutor::testCompletion(std::shared_ptr<TensorNetworkReq> tn_req)
{
 //Test completion on all GPUs:
 bool all_completed = true;
 const int num_gpus = gpu_attr_.size();
 for(int gpu = 0; gpu < num_gpus; ++gpu){
  const auto gpu_id = gpu_attr_[gpu].first;
  HANDLE_CUDA_ERROR(cudaSetDevice(gpu_id));
  cudaError_t cuda_error = cudaEventQuery(tn_req->gpu_data_out_finish[gpu]);
  if(cuda_error != cudaErrorNotReady){
   if(tn_req->memory_window_ptr[gpu] != nullptr){
    mem_pool_[gpu].releaseMemory(tn_req->memory_window_ptr[gpu]);
    tn_req->memory_window_ptr[gpu] = nullptr;
   }
  }else{
   all_completed = false;
  }
 }
 if(all_completed){
#ifdef MPI_ENABLED
  //Global output reduction across all participating MPI processes:
  if(tn_req->num_procs > 1){
   const auto out_elem_type = tn_req->network->getTensor(0)->getElementType();
   const auto out_tens_hash = tn_req->network->getTensor(0)->getTensorHash();
   const auto & descr = tn_req->tensor_descriptors[out_tens_hash];
   auto & mpi_comm = tn_req->comm.getRef<MPI_Comm>();
   int errc = MPI_SUCCESS;
   switch(out_elem_type){
    case TensorElementType::REAL32:
     errc = MPI_Allreduce(MPI_IN_PLACE,descr.src_ptr,descr.volume,MPI_FLOAT,MPI_SUM,mpi_comm);
     break;
    case TensorElementType::REAL64:
     errc = MPI_Allreduce(MPI_IN_PLACE,descr.src_ptr,descr.volume,MPI_DOUBLE,MPI_SUM,mpi_comm);
     break;
    case TensorElementType::COMPLEX32:
     errc = MPI_Allreduce(MPI_IN_PLACE,descr.src_ptr,descr.volume,MPI_CXX_FLOAT_COMPLEX,MPI_SUM,mpi_comm);
     break;
    case TensorElementType::COMPLEX64:
     errc = MPI_Allreduce(MPI_IN_PLACE,descr.src_ptr,descr.volume,MPI_CXX_DOUBLE_COMPLEX,MPI_SUM,mpi_comm);
     break;
    default:
     fatal_error("#ERROR(exatn::CuQuantumExecutor::testCompletion): Invalid tensor element type!");
   }
   assert(errc == MPI_SUCCESS);
  }
#endif
  tn_req->exec_status = TensorNetworkQueue::ExecStat::Completed;
 }
 return;
}


ExecutionTimings ExecutionTimings::computeAverage(const std::vector<ExecutionTimings> & timings)
{
 ExecutionTimings result_timings = std::accumulate(timings.cbegin(),timings.cend(),
                                    ExecutionTimings{0.0f,0.0f,0.0f,0.0f},
                                    [](const ExecutionTimings & first, const ExecutionTimings & second){
                                     return ExecutionTimings{first.prepare + second.prepare,
                                                             first.data_in + second.data_in,
                                                             first.data_out + second.data_out,
                                                             first.compute + second.compute};
                                    });
 if(!timings.empty()){
  const float num_elems = static_cast<float>(timings.size());
  result_timings.prepare /= num_elems;
  result_timings.data_in /= num_elems;
  result_timings.data_out /= num_elems;
  result_timings.compute /= num_elems;
 }
 return std::move(result_timings);
}


ExecutionTimings ExecutionTimings::computeWorst(const std::vector<ExecutionTimings> & timings)
{
 ExecutionTimings result_timings{0.0f,0.0f,0.0f,0.0f};
 if(!timings.empty()){
  result_timings = std::accumulate(timings.cbegin()+1,timings.cend(),
                                   timings[0],
                                   [](const ExecutionTimings & first, const ExecutionTimings & second){
                                    return ExecutionTimings{std::max(first.prepare,second.prepare),
                                                            std::max(first.data_in,second.data_in),
                                                            std::max(first.data_out,second.data_out),
                                                            std::max(first.compute,second.compute)};
                                   });
 }
 return std::move(result_timings);
}


ExecutionTimings ExecutionTimings::computeBest(const std::vector<ExecutionTimings> & timings)
{
 ExecutionTimings result_timings{0.0f,0.0f,0.0f,0.0f};
 if(!timings.empty()){
  result_timings = std::accumulate(timings.cbegin()+1,timings.cend(),
                                   timings[0],
                                   [](const ExecutionTimings & first, const ExecutionTimings & second){
                                    return ExecutionTimings{std::min(first.prepare,second.prepare),
                                                            std::min(first.data_in,second.data_in),
                                                            std::min(first.data_out,second.data_out),
                                                            std::min(first.compute,second.compute)};
                                   });
 }
 return std::move(result_timings);
}

} //namespace runtime
} //namespace exatn

#endif //CUQUANTUM
