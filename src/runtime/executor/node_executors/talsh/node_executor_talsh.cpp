/** ExaTN:: Tensor Runtime: Tensor graph node executor: Talsh
REVISION: 2020/08/10

Copyright (C) 2018-2020 Dmitry Lyakh, Tiffany Mintz, Alex McCaskey
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle)
**/

#include "node_executor_talsh.hpp"

#ifdef MPI_ENABLED
#include "mpi.h"
#endif

#include <complex>
#include <limits>
#include <mutex>

#include <cstdlib>
#include <cassert>

namespace exatn {
namespace runtime {

bool TalshNodeExecutor::talsh_initialized_ = false;
int TalshNodeExecutor::talsh_node_exec_count_ = 0;

std::mutex talsh_init_lock;


#ifdef MPI_ENABLED
inline MPI_Datatype get_mpi_tensor_element_kind(int talsh_data_kind)
{
 MPI_Datatype mpi_data_kind;
 switch(talsh_data_kind){
 case talsh::REAL32: mpi_data_kind = MPI_REAL; break;
 case talsh::REAL64: mpi_data_kind = MPI_DOUBLE_PRECISION; break;
 case talsh::COMPLEX32: mpi_data_kind = MPI_COMPLEX; break;
 case talsh::COMPLEX64: mpi_data_kind = MPI_DOUBLE_COMPLEX; break;
 default:
  std::cout << "#FATAL(exatn::runtime::TalshNodeExecutor): Unknown TAL-SH data kind: "
            << talsh_data_kind << std::endl;
  assert(false);
 }
 return mpi_data_kind;
}
#endif


void TalshNodeExecutor::initialize(const ParamConf & parameters)
{
#ifndef NDEBUG
  const bool debugging = true;
#else
  const bool debugging = false;
#endif
 talsh_init_lock.lock();
 if(!talsh_initialized_){
  std::size_t host_mem_buffer_size = DEFAULT_MEM_BUFFER_SIZE;
  int64_t provided_buf_size = 0;
  if(parameters.getParameter("host_memory_buffer_size",&provided_buf_size))
   host_mem_buffer_size = provided_buf_size;
  auto error_code = talsh::initialize(&host_mem_buffer_size);
  if(error_code == TALSH_SUCCESS){
   talsh_host_mem_buffer_size_.store(host_mem_buffer_size);
   if (debugging) std::cout << "#DEBUG(exatn::runtime::TalshNodeExecutor): TAL-SH initialized with Host buffer size of " <<
    talsh_host_mem_buffer_size_ << " bytes" << std::endl << std::flush; //debug
   talsh_initialized_ = true;
  }else{
   std::cerr << "#FATAL(exatn::runtime::TalshNodeExecutor): Unable to initialize TAL-SH!" << std::endl;
   assert(false);
  }
 }
 ++talsh_node_exec_count_;
 talsh_init_lock.unlock();
 return;
}


std::size_t TalshNodeExecutor::getMemoryBufferSize() const
{
 std::size_t buf_size = 0;
 while(buf_size == 0) buf_size = talsh_host_mem_buffer_size_.load();
 return buf_size;
}


TalshNodeExecutor::~TalshNodeExecutor()
{
#ifndef NDEBUG
  const bool debugging = true;
#else
  const bool debugging = false;
#endif
 auto synced = sync(true); assert(synced);
 talsh_init_lock.lock();
 --talsh_node_exec_count_;
 if(talsh_initialized_ && talsh_node_exec_count_ == 0){
  tasks_.clear();
  tensors_.clear();
  talsh::printStatistics();
  auto error_code = talsh::shutdown();
  if(error_code == TALSH_SUCCESS){
   if (debugging) std::cout << "#DEBUG(exatn::runtime::TalshNodeExecutor): TAL-SH shut down" << std::endl << std::flush;
   talsh_initialized_ = false;
  }else{
   std::cerr << "#FATAL(exatn::runtime::TalshNodeExecutor): Unable to shut down TAL-SH!" << std::endl;
   assert(false);
  }
 }
 talsh_init_lock.unlock();
}


TalshNodeExecutor::TensorImpl::TensorImpl(const std::vector<std::size_t> & full_offsets,
                                          const std::vector<DimExtent> & full_extents,
                                          const std::vector<std::size_t> & reduced_offsets,
                                          const std::vector<int> & reduced_extents,
                                          int data_kind):
 talsh_tensor(new talsh::Tensor(reduced_offsets,reduced_extents,data_kind,talsh_tens_no_init)),
 full_base_offsets(full_offsets), reduced_base_offsets(reduced_offsets),
 stored_shape(nullptr), full_shape_is_on(false)
{
 auto errc = tensShape_create(&stored_shape); assert(errc == TALSH_SUCCESS);
 int full_rank = full_extents.size();
 int dims[full_rank];
 for(int i = 0; i < full_rank; ++i){
  if(full_extents[i] > static_cast<DimExtent>(std::numeric_limits<int>::max())){
   std::cout << "#FATAL(exatn::runtime::TalshNodeExecutor): CREATE: Tensor dimension extent exceeds max int: "
             << full_extents[i] << std::endl << std::flush;
   assert(false);
  }
  dims[i] = full_extents[i];
 }
 errc = tensShape_construct(stored_shape,NOPE,full_rank,dims); assert(errc == TALSH_SUCCESS);
}


TalshNodeExecutor::TensorImpl::TensorImpl(TalshNodeExecutor::TensorImpl && other) noexcept:
 talsh_tensor(std::move(other.talsh_tensor)),
 full_base_offsets(std::move(other.full_base_offsets)),
 reduced_base_offsets(std::move(other.reduced_base_offsets)),
 stored_shape(other.stored_shape), full_shape_is_on(other.full_shape_is_on)
{
 other.stored_shape = nullptr;
}


TalshNodeExecutor::TensorImpl & TalshNodeExecutor::TensorImpl::operator=(TalshNodeExecutor::TensorImpl && other) noexcept
{
 if(this != &other){
  if(stored_shape != nullptr){
   resetTensorShapeToReduced();
   auto errc = tensShape_destroy(stored_shape); assert(errc == TALSH_SUCCESS);
  }
  stored_shape = other.stored_shape;
  other.stored_shape = nullptr;
  full_base_offsets = std::move(other.full_base_offsets);
  reduced_base_offsets = std::move(other.reduced_base_offsets);
  talsh_tensor = std::move(other.talsh_tensor);
  full_shape_is_on = other.full_shape_is_on;
 }
 return *this;
}


TalshNodeExecutor::TensorImpl::~TensorImpl()
{
 if(stored_shape != nullptr){
  resetTensorShapeToReduced();
  auto errc = tensShape_destroy(stored_shape); assert(errc == TALSH_SUCCESS);
  stored_shape = nullptr;
 }
}


void TalshNodeExecutor::TensorImpl::resetTensorShapeToFull()
{
 if(!full_shape_is_on){
  talsh_tensor->resetDimOffsets(full_base_offsets);
  auto * talsh_tens = talsh_tensor->getTalshTensorPtr();
  talsh_tens_shape_t * current_shape = talsh_tens->shape_p;
  assert(current_shape != nullptr && stored_shape != nullptr);
  talsh_tens->shape_p = stored_shape;
  stored_shape = current_shape;
  full_shape_is_on = true;
 }
 return;
}


void TalshNodeExecutor::TensorImpl::resetTensorShapeToReduced()
{
 if(full_shape_is_on){
  talsh_tensor->resetDimOffsets(reduced_base_offsets);
  auto * talsh_tens = talsh_tensor->getTalshTensorPtr();
  talsh_tens_shape_t * current_shape = talsh_tens->shape_p;
  assert(current_shape != nullptr && stored_shape != nullptr);
  talsh_tens->shape_p = stored_shape;
  stored_shape = current_shape;
  full_shape_is_on = false;
 }
 return;
}


int TalshNodeExecutor::execute(numerics::TensorOpCreate & op,
                               TensorOpExecHandle * exec_handle)
{
 assert(op.isSet());

 const auto & tensor = *(op.getTensorOperand(0));
 const auto & tensor_signature = tensor.getSignature();
 const auto full_tensor_rank = tensor.getRank();
 const auto tensor_hash = tensor.getTensorHash();
 //Get tensor dimension extents:
 const auto & dim_extents = tensor.getDimExtents();
 //Get tensor dimension base offsets:
 std::vector<std::size_t> offsets(full_tensor_rank);
 for(int i = 0; i < full_tensor_rank; ++i){
  auto space_id = tensor_signature.getDimSpaceId(i);
  auto subspace_id = tensor_signature.getDimSubspaceId(i);
  if(space_id == SOME_SPACE){
   offsets[i] = static_cast<std::size_t>(subspace_id);
  }else{
   const auto * subspace = getSpaceRegister()->getSubspace(space_id,subspace_id);
   offsets[i] = static_cast<std::size_t>(subspace->getLowerBound());
  }
 }
 //Remove tensor dimensions of extent 1:
 unsigned int tensor_rank = 0;
 for(int i = 0; i < full_tensor_rank; ++i){
  if(dim_extents[i] > 1){
   if(dim_extents[i] <= static_cast<DimExtent>(std::numeric_limits<int>::max())){
    ++tensor_rank;
   }else{
    std::cout << "#ERROR(exatn::runtime::node_executor_talsh): CREATE: Tensor dimension extent exceeds max int: "
              << dim_extents[i] << std::endl << std::flush;
    assert(false);
   }
  }
 }
 std::vector<int> extents(tensor_rank);
 std::vector<std::size_t> bases(tensor_rank);
 tensor_rank = 0;
 for(int i = 0; i < full_tensor_rank; ++i){
  if(dim_extents[i] > 1){
   extents[tensor_rank] = static_cast<int>(dim_extents[i]);
   bases[tensor_rank++] = offsets[i];
  }
 }
 //Get tensor data kind:
 auto data_kind = get_talsh_tensor_element_kind(op.getTensorElementType());
 //Construct the TAL-SH tensor implementation:
 auto res = tensors_.emplace(std::make_pair(tensor_hash,TensorImpl(offsets,dim_extents,bases,extents,data_kind)));
 if(res.second){
  if(res.first->second.talsh_tensor->isEmpty()){ //tensor has not been allocated memory due to its temporary shortage
   tensors_.erase(res.first);
   return TRY_LATER;
  }
  //std::cout << "#DEBUG(exatn::runtime::node_executor_talsh): New tensor " << tensor.getName()
  //          << " emplaced with hash " << tensor_hash << std::endl;
 }else{
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): CREATE: Attempt to create the same tensor twice: " << std::endl;
  tensor.printIt();
  assert(false);
 }
 *exec_handle = op.getId();
 return 0;
}


int TalshNodeExecutor::execute(numerics::TensorOpDestroy & op,
                               TensorOpExecHandle * exec_handle)
{
 assert(op.isSet());
 if(!finishPrefetching(op)) return TRY_LATER;

 const auto & tensor = *(op.getTensorOperand(0));
 const auto tensor_hash = tensor.getTensorHash();
 auto iter = tensors_.find(tensor_hash);
 if(iter != tensors_.end()){
  //Complete an active tensor image eviction, if any:
  auto eviction = evictions_.find(iter->second.talsh_tensor.get());
  if(eviction != evictions_.end()){
   auto snc = eviction->second->wait();
   evictions_.erase(eviction);
  }
  //Tensor destruction procedure:
  bool in_use = tensorIsCurrentlyInUse(iter->second.talsh_tensor.get());
  if(!in_use){
   //Evict the tensor from device caches:
   for(int dev = 0; dev < DEV_MAX; ++dev){
    auto cached = accel_cache_[dev].find(iter->second.talsh_tensor.get());
    if(cached != accel_cache_[dev].end()) accel_cache_[dev].erase(cached);
   }
   //Move tensor image to Host:
   auto synced = iter->second.talsh_tensor->sync(DEV_HOST,0,nullptr,true); assert(synced);
   //Destroy the tensor:
   iter->second.resetTensorShapeToReduced();
   tensors_.erase(iter);
   //std::cout << "#DEBUG(exatn::runtime::node_executor_talsh): Tensor " << tensor.getName()
   //          << " erased with hash " << tensor_hash << std::endl;
  }else{
   std::cout << "#ERROR(exatn::runtime::node_executor_talsh): DESTROY: Attempt to destroy an active tensor:" << std::endl;
   tensor.printIt();
   assert(false);
  }
 }else{
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): DESTROY: Attempt to destroy non-existing tensor:" << std::endl;
  tensor.printIt();
  assert(false);
 }
 *exec_handle = op.getId();
 return 0;
}


int TalshNodeExecutor::execute(numerics::TensorOpTransform & op,
                               TensorOpExecHandle * exec_handle)
{
 assert(op.isSet());
 if(!finishPrefetching(op)) return TRY_LATER;

 const auto & tensor = *(op.getTensorOperand(0));
 const auto tensor_hash = tensor.getTensorHash();
 auto tens_pos = tensors_.find(tensor_hash);
 if(tens_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): TRANSFORM: Tensor operand 0 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 tens_pos->second.resetTensorShapeToFull();
 auto & tens = *(tens_pos->second.talsh_tensor);
 auto synced = tens.sync(DEV_HOST,0,nullptr,true); assert(synced);
 int error_code = op.apply(tens); //synchronous user-defined Host operation
 *exec_handle = op.getId();
 return error_code;
}


int TalshNodeExecutor::execute(numerics::TensorOpSlice & op,
                               TensorOpExecHandle * exec_handle)
{
 assert(op.isSet());
 if(!finishPrefetching(op)) return TRY_LATER;

 const auto & tensor0 = *(op.getTensorOperand(0));
 const auto tensor0_hash = tensor0.getTensorHash();
 auto tens0_pos = tensors_.find(tensor0_hash);
 if(tens0_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): SLICE: Tensor operand 0 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 tens0_pos->second.resetTensorShapeToFull();
 auto & tens0 = *(tens0_pos->second.talsh_tensor);

 const auto & tensor1 = *(op.getTensorOperand(1));
 const auto tensor1_hash = tensor1.getTensorHash();
 auto tens1_pos = tensors_.find(tensor1_hash);
 if(tens1_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): SLICE: Tensor operand 1 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 tens1_pos->second.resetTensorShapeToFull();
 auto & tens1 = *(tens1_pos->second.talsh_tensor);

 *exec_handle = op.getId();
 auto task_res = tasks_.emplace(std::make_pair(*exec_handle,
                                std::make_shared<talsh::TensorTask>()));
 if(!task_res.second){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): SLICE: Attempt to execute the same operation twice: " << std::endl;
  op.printIt();
  assert(false);
 }

 const auto & slice_signature = tensor0.getSignature();
 const auto slice_rank = slice_signature.getRank();
 std::vector<int> offsets(slice_rank);
 for(unsigned int i = 0; i < slice_rank; ++i){
  auto space_id = slice_signature.getDimSpaceId(i);
  auto subspace_id = slice_signature.getDimSubspaceId(i);
  if(space_id == SOME_SPACE){
   if(subspace_id > std::numeric_limits<int>::max()){
    std::cout << "#ERROR(exatn::runtime::node_executor_talsh): SLICE: Integer (int) overflow in offsets" << std::endl;
    assert(false);
   }
   offsets[i] = static_cast<int>(subspace_id);
  }else{
   const auto * subspace = getSpaceRegister()->getSubspace(space_id,subspace_id);
   auto lower_bound = subspace->getLowerBound();
   if(lower_bound > std::numeric_limits<int>::max()){
    std::cout << "#ERROR(exatn::runtime::node_executor_talsh): SLICE: Integer (int) overflow in offsets" << std::endl;
    assert(false);
   }
   offsets[i] = static_cast<int>(lower_bound);
  }
 }

 auto error_code = tens1.extractSlice((task_res.first)->second.get(),
                                      tens0,
                                      offsets,
                                      DEV_HOST,0);

 return error_code;
}


int TalshNodeExecutor::execute(numerics::TensorOpInsert & op,
                               TensorOpExecHandle * exec_handle)
{
 assert(op.isSet());
 if(!finishPrefetching(op)) return TRY_LATER;

 const auto & tensor0 = *(op.getTensorOperand(0));
 const auto tensor0_hash = tensor0.getTensorHash();
 auto tens0_pos = tensors_.find(tensor0_hash);
 if(tens0_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): INSERT: Tensor operand 0 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 tens0_pos->second.resetTensorShapeToFull();
 auto & tens0 = *(tens0_pos->second.talsh_tensor);

 const auto & tensor1 = *(op.getTensorOperand(1));
 const auto tensor1_hash = tensor1.getTensorHash();
 auto tens1_pos = tensors_.find(tensor1_hash);
 if(tens1_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): INSERT: Tensor operand 1 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 tens1_pos->second.resetTensorShapeToFull();
 auto & tens1 = *(tens1_pos->second.talsh_tensor);

 *exec_handle = op.getId();
 auto task_res = tasks_.emplace(std::make_pair(*exec_handle,
                                std::make_shared<talsh::TensorTask>()));
 if(!task_res.second){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): INSERT: Attempt to execute the same operation twice: " << std::endl;
  op.printIt();
  assert(false);
 }

 const auto & slice_signature = tensor1.getSignature();
 const auto slice_rank = slice_signature.getRank();
 std::vector<int> offsets(slice_rank);
 for(unsigned int i = 0; i < slice_rank; ++i){
  auto space_id = slice_signature.getDimSpaceId(i);
  auto subspace_id = slice_signature.getDimSubspaceId(i);
  if(space_id == SOME_SPACE){
   if(subspace_id > std::numeric_limits<int>::max()){
    std::cout << "#ERROR(exatn::runtime::node_executor_talsh): INSERT: Integer (int) overflow in offsets" << std::endl;
    assert(false);
   }
   offsets[i] = static_cast<int>(subspace_id);
  }else{
   const auto * subspace = getSpaceRegister()->getSubspace(space_id,subspace_id);
   auto lower_bound = subspace->getLowerBound();
   if(lower_bound > std::numeric_limits<int>::max()){
    std::cout << "#ERROR(exatn::runtime::node_executor_talsh): INSERT: Integer (int) overflow in offsets" << std::endl;
    assert(false);
   }
   offsets[i] = static_cast<int>(lower_bound);
  }
 }

 auto error_code = tens0.insertSlice((task_res.first)->second.get(),
                                     tens1,
                                     offsets,
                                     DEV_HOST,0);

 return error_code;
}


int TalshNodeExecutor::execute(numerics::TensorOpAdd & op,
                               TensorOpExecHandle * exec_handle)
{
 assert(op.isSet());
 if(!finishPrefetching(op)) return TRY_LATER;

 const auto & tensor0 = *(op.getTensorOperand(0));
 const auto tensor0_hash = tensor0.getTensorHash();
 auto tens0_pos = tensors_.find(tensor0_hash);
 if(tens0_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): ADD: Tensor operand 0 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 tens0_pos->second.resetTensorShapeToReduced();
 auto & tens0 = *(tens0_pos->second.talsh_tensor);

 const auto & tensor1 = *(op.getTensorOperand(1));
 const auto tensor1_hash = tensor1.getTensorHash();
 auto tens1_pos = tensors_.find(tensor1_hash);
 if(tens1_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): ADD: Tensor operand 1 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 tens1_pos->second.resetTensorShapeToReduced();
 auto & tens1 = *(tens1_pos->second.talsh_tensor);

 *exec_handle = op.getId();
 auto task_res = tasks_.emplace(std::make_pair(*exec_handle,
                                std::make_shared<talsh::TensorTask>()));
 if(!task_res.second){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): ADD: Attempt to execute the same operation twice: " << std::endl;
  op.printIt();
  assert(false);
 }

 auto error_code = tens0.accumulate((task_res.first)->second.get(),
                                    op.getIndexPatternReduced(),
                                    tens1,
                                    DEV_DEFAULT,DEV_DEFAULT,
                                    op.getScalar(0));
 if(error_code == DEVICE_UNABLE || error_code == TALSH_NOT_AVAILABLE || error_code == TALSH_NOT_IMPLEMENTED){
  (task_res.first)->second->clean();
  error_code = tens0.accumulate((task_res.first)->second.get(),
                                op.getIndexPatternReduced(),
                                tens1,
                                DEV_HOST,0,
                                op.getScalar(0));
 }else if(error_code == TRY_LATER){
  std::size_t total_tensor_size = tensor0.getSize() + tensor1.getSize();
  auto evicting = evictMovedTensors(talsh::determineOptimalDevice(tens0,tens1),total_tensor_size);
 }
 return error_code;
}


int TalshNodeExecutor::execute(numerics::TensorOpContract & op,
                               TensorOpExecHandle * exec_handle)
{
 assert(op.isSet());
 if(!finishPrefetching(op)) return TRY_LATER;

 const auto & tensor0 = *(op.getTensorOperand(0));
 const auto tensor0_hash = tensor0.getTensorHash();
 auto tens0_pos = tensors_.find(tensor0_hash);
 if(tens0_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): CONTRACT: Tensor operand 0 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 tens0_pos->second.resetTensorShapeToReduced();
 auto & tens0 = *(tens0_pos->second.talsh_tensor);

 const auto & tensor1 = *(op.getTensorOperand(1));
 const auto tensor1_hash = tensor1.getTensorHash();
 auto tens1_pos = tensors_.find(tensor1_hash);
 if(tens1_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): CONTRACT: Tensor operand 1 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 tens1_pos->second.resetTensorShapeToReduced();
 auto & tens1 = *(tens1_pos->second.talsh_tensor);

 const auto & tensor2 = *(op.getTensorOperand(2));
 const auto tensor2_hash = tensor2.getTensorHash();
 auto tens2_pos = tensors_.find(tensor2_hash);
 if(tens2_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): CONTRACT: Tensor operand 2 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 tens2_pos->second.resetTensorShapeToReduced();
 auto & tens2 = *(tens2_pos->second.talsh_tensor);

 *exec_handle = op.getId();
 auto task_res = tasks_.emplace(std::make_pair(*exec_handle,
                                std::make_shared<talsh::TensorTask>()));
 if(!task_res.second){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): CONTRACT: Attempt to execute the same operation twice: " << std::endl;
  op.printIt();
  assert(false);
 }

 //std::cout << "#DEBUG(exatn::runtime::node_executor_talsh): Tensor contraction " << op.getIndexPattern() << std::endl; //debug
 auto error_code = tens0.contractAccumulate((task_res.first)->second.get(),
                                            op.getIndexPatternReduced(),
                                            tens1,tens2,
                                            DEV_DEFAULT,DEV_DEFAULT,
                                            op.getScalar(0));
 if(error_code == DEVICE_UNABLE){ //use out-of-core version if tensor contraction does not fit in GPU
  //std::cout << "#DEBUG(exatn::runtime::node_executor_talsh): CONTRACT: Redirected to XL\n" << std::flush; //debug
  (task_res.first)->second->clean();
  bool synced = sync(true); //completes all active tasks, prefetches and evictions
  bool evicting = evictMovedTensors(DEV_DEFAULT,0); //evict all cached tensors from all accelerators
  if(evicting) synced = synced && sync(true); //completes all evictions
  task_res = tasks_.emplace(std::make_pair(*exec_handle,
                            std::make_shared<talsh::TensorTask>()));
  if(synced){
   error_code = tens0.contractAccumulateXL((task_res.first)->second.get(),
                                           op.getIndexPatternReduced(),
                                           tens1,tens2,
                                           DEV_DEFAULT,DEV_DEFAULT,
                                           op.getScalar(0));
  }else{
   error_code = tens0.contractAccumulate((task_res.first)->second.get(),
                                         op.getIndexPatternReduced(),
                                         tens1,tens2,
                                         DEV_HOST,0,
                                         op.getScalar(0));
  }
 }else if(error_code == TALSH_NOT_AVAILABLE || error_code == TALSH_NOT_IMPLEMENTED){
  (task_res.first)->second->clean();
  error_code = tens0.contractAccumulate((task_res.first)->second.get(),
                                         op.getIndexPatternReduced(),
                                         tens1,tens2,
                                         DEV_HOST,0,
                                         op.getScalar(0));
 }else if(error_code == TRY_LATER){
  //(task_res.first)->second->clean();
  std::size_t total_tensor_size = tensor0.getSize() + tensor1.getSize() + tensor2.getSize();
  bool evicting = evictMovedTensors(talsh::determineOptimalDevice(tens0,tens1,tens2),total_tensor_size);
 }
 return error_code;
}


int TalshNodeExecutor::execute(numerics::TensorOpDecomposeSVD3 & op,
                               TensorOpExecHandle * exec_handle)
{
 assert(op.isSet());
 if(!finishPrefetching(op)) return TRY_LATER;

 const auto & tensor0 = *(op.getTensorOperand(0));
 const auto tensor0_hash = tensor0.getTensorHash();
 auto tens0_pos = tensors_.find(tensor0_hash);
 if(tens0_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): DECOMPOSE_SVD3: Tensor operand 0 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 tens0_pos->second.resetTensorShapeToReduced();
 auto & tens0 = *(tens0_pos->second.talsh_tensor);

 const auto & tensor1 = *(op.getTensorOperand(1));
 const auto tensor1_hash = tensor1.getTensorHash();
 auto tens1_pos = tensors_.find(tensor1_hash);
 if(tens1_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): DECOMPOSE_SVD3: Tensor operand 1 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 tens1_pos->second.resetTensorShapeToReduced();
 auto & tens1 = *(tens1_pos->second.talsh_tensor);

 const auto & tensor2 = *(op.getTensorOperand(2));
 const auto tensor2_hash = tensor2.getTensorHash();
 auto tens2_pos = tensors_.find(tensor2_hash);
 if(tens2_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): DECOMPOSE_SVD3: Tensor operand 2 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 tens2_pos->second.resetTensorShapeToReduced();
 auto & tens2 = *(tens2_pos->second.talsh_tensor);

 const auto & tensor3 = *(op.getTensorOperand(3));
 const auto tensor3_hash = tensor3.getTensorHash();
 auto tens3_pos = tensors_.find(tensor3_hash);
 if(tens3_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): DECOMPOSE_SVD3: Tensor operand 3 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 tens3_pos->second.resetTensorShapeToReduced();
 auto & tens3 = *(tens3_pos->second.talsh_tensor);

 *exec_handle = op.getId();
 auto task_res = tasks_.emplace(std::make_pair(*exec_handle,
                                std::make_shared<talsh::TensorTask>()));
 if(!task_res.second){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): DECOMPOSE_SVD3: Attempt to execute the same operation twice: " << std::endl;
  op.printIt();
  assert(false);
 }

 auto error_code = tens3.decomposeSVD((task_res.first)->second.get(),
                                      op.getIndexPatternReduced(),
                                      tens0,tens1,tens2,
                                      DEV_HOST,0);
 return error_code;
}


int TalshNodeExecutor::execute(numerics::TensorOpDecomposeSVD2 & op,
                               TensorOpExecHandle * exec_handle)
{
 assert(op.isSet());
 if(!finishPrefetching(op)) return TRY_LATER;

 const auto & tensor0 = *(op.getTensorOperand(0));
 const auto tensor0_hash = tensor0.getTensorHash();
 auto tens0_pos = tensors_.find(tensor0_hash);
 if(tens0_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): DECOMPOSE_SVD2: Tensor operand 0 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 tens0_pos->second.resetTensorShapeToReduced();
 auto & tens0 = *(tens0_pos->second.talsh_tensor);

 const auto & tensor1 = *(op.getTensorOperand(1));
 const auto tensor1_hash = tensor1.getTensorHash();
 auto tens1_pos = tensors_.find(tensor1_hash);
 if(tens1_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): DECOMPOSE_SVD2: Tensor operand 1 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 tens1_pos->second.resetTensorShapeToReduced();
 auto & tens1 = *(tens1_pos->second.talsh_tensor);

 const auto & tensor2 = *(op.getTensorOperand(2));
 const auto tensor2_hash = tensor2.getTensorHash();
 auto tens2_pos = tensors_.find(tensor2_hash);
 if(tens2_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): DECOMPOSE_SVD2: Tensor operand 2 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 tens2_pos->second.resetTensorShapeToReduced();
 auto & tens2 = *(tens2_pos->second.talsh_tensor);

 *exec_handle = op.getId();
 auto task_res = tasks_.emplace(std::make_pair(*exec_handle,
                                std::make_shared<talsh::TensorTask>()));
 if(!task_res.second){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): DECOMPOSE_SVD2: Attempt to execute the same operation twice: " << std::endl;
  op.printIt();
  assert(false);
 }

 auto error_code = tens2.decomposeSVDLR((task_res.first)->second.get(),
                                        op.getIndexPatternReduced(),
                                        tens0,tens1,
                                        DEV_HOST,0);
 return error_code;
}


int TalshNodeExecutor::execute(numerics::TensorOpOrthogonalizeSVD & op,
                               TensorOpExecHandle * exec_handle)
{
 assert(op.isSet());
 if(!finishPrefetching(op)) return TRY_LATER;

 const auto & tensor0 = *(op.getTensorOperand(0));
 const auto tensor0_hash = tensor0.getTensorHash();
 auto tens0_pos = tensors_.find(tensor0_hash);
 if(tens0_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): ORTHOGONALIZE_SVD: Tensor operand 0 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 tens0_pos->second.resetTensorShapeToReduced();
 auto & tens0 = *(tens0_pos->second.talsh_tensor);

 *exec_handle = op.getId();
 auto task_res = tasks_.emplace(std::make_pair(*exec_handle,
                                std::make_shared<talsh::TensorTask>()));
 if(!task_res.second){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): ORTHOGONALIZE_SVD: Attempt to execute the same operation twice: " << std::endl;
  op.printIt();
  assert(false);
 }

 auto error_code = tens0.orthogonalizeSVD((task_res.first)->second.get(),
                                          op.getIndexPatternReduced(),
                                          DEV_HOST,0);
 return error_code;
}


int TalshNodeExecutor::execute(numerics::TensorOpOrthogonalizeMGS & op,
                               TensorOpExecHandle * exec_handle)
{
 assert(op.isSet());
 if(!finishPrefetching(op)) return TRY_LATER;

 const auto & tensor0 = *(op.getTensorOperand(0));
 const auto tensor0_hash = tensor0.getTensorHash();
 auto tens0_pos = tensors_.find(tensor0_hash);
 if(tens0_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): ORTHOGONALIZE_MGS: Tensor operand 0 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 tens0_pos->second.resetTensorShapeToReduced();
 auto & tens0 = *(tens0_pos->second.talsh_tensor);

 *exec_handle = op.getId();
 auto task_res = tasks_.emplace(std::make_pair(*exec_handle,
                                std::make_shared<talsh::TensorTask>()));
 if(!task_res.second){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): ORTHOGONALIZE_MGS: Attempt to execute the same operation twice: " << std::endl;
  op.printIt();
  assert(false);
 }

 auto error_code = 0;
 /*
 auto error_code = tens0.orthogonalizeMGS((task_res.first)->second.get(),
                                          op.getIndexPatternReduced(),
                                          DEV_HOST,0);
 */
 return error_code;
}


int TalshNodeExecutor::execute(numerics::TensorOpBroadcast & op,
                               TensorOpExecHandle * exec_handle)
{
 assert(op.isSet());
 if(!finishPrefetching(op)) return TRY_LATER;

 const auto & tensor = *(op.getTensorOperand(0));
 const auto tensor_hash = tensor.getTensorHash();
 auto tens_pos = tensors_.find(tensor_hash);
 if(tens_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): BROADCAST: Tensor operand 0 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 tens_pos->second.resetTensorShapeToReduced();
 auto & tens = *(tens_pos->second.talsh_tensor);

 *exec_handle = op.getId();

 int error_code = 0;
#ifdef MPI_ENABLED
 auto synced = tens.sync(DEV_HOST,0,nullptr,true); assert(synced);
 float * tens_body_r4 = nullptr;
 double * tens_body_r8 = nullptr;
 std::complex<float> * tens_body_c4 = nullptr;
 std::complex<double> * tens_body_c8 = nullptr;
 bool access_granted = false;
 int tens_elem_type = tens.getElementType();
 switch(tens_elem_type){
  case(talsh::REAL32): access_granted = tens.getDataAccessHost(&tens_body_r4); break;
  case(talsh::REAL64): access_granted = tens.getDataAccessHost(&tens_body_r8); break;
  case(talsh::COMPLEX32): access_granted = tens.getDataAccessHost(&tens_body_c4); break;
  case(talsh::COMPLEX64): access_granted = tens.getDataAccessHost(&tens_body_c8); break;
  default:
   std::cout << "#ERROR(exatn::runtime::node_executor_talsh): BROADCAST: Unknown TAL-SH data kind: "
             << tens_elem_type << std::endl;
   op.printIt();
   assert(false);
 }
 if(access_granted){
  auto mpi_data_kind = get_mpi_tensor_element_kind(tens_elem_type);
  auto communicator = *(op.getMPICommunicator().get<MPI_Comm>());
  int root_rank = op.getRootRank();
  std::size_t tens_volume = tens.getVolume();
  int chunk = std::numeric_limits<int>::max();
  for(std::size_t base = 0; base < tens_volume; base += chunk){
   int count = std::min(chunk,static_cast<int>(tens_volume-base));
   switch(tens_elem_type){
    case(talsh::REAL32):
     assert(tens_body_r4 != nullptr);
     error_code = MPI_Bcast((void*)(&(tens_body_r4[base])),count,mpi_data_kind,root_rank,communicator);
     break;
    case(talsh::REAL64):
     assert(tens_body_r8 != nullptr);
     error_code = MPI_Bcast((void*)(&(tens_body_r8[base])),count,mpi_data_kind,root_rank,communicator);
     break;
    case(talsh::COMPLEX32):
     assert(tens_body_c4 != nullptr);
     error_code = MPI_Bcast((void*)(&(tens_body_c4[base])),count,mpi_data_kind,root_rank,communicator);
     break;
    case(talsh::COMPLEX64):
     assert(tens_body_c8 != nullptr);
     error_code = MPI_Bcast((void*)(&(tens_body_c8[base])),count,mpi_data_kind,root_rank,communicator);
     break;
   }
   if(error_code != MPI_SUCCESS) break;
  }
 }else{
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): BROADCAST: Unable to get access to the tensor body!" << std::endl;
  op.printIt();
  assert(false);
 }
#endif
 return error_code;
}


int TalshNodeExecutor::execute(numerics::TensorOpAllreduce & op,
                               TensorOpExecHandle * exec_handle)
{
 assert(op.isSet());
 if(!finishPrefetching(op)) return TRY_LATER;

 const auto & tensor = *(op.getTensorOperand(0));
 const auto tensor_hash = tensor.getTensorHash();
 auto tens_pos = tensors_.find(tensor_hash);
 if(tens_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): ALLREDUCE: Tensor operand 0 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 tens_pos->second.resetTensorShapeToReduced();
 auto & tens = *(tens_pos->second.talsh_tensor);

 *exec_handle = op.getId();

 int error_code = 0;
#ifdef MPI_ENABLED
 auto synced = tens.sync(DEV_HOST,0,nullptr,true); assert(synced);
 float * tens_body_r4 = nullptr;
 double * tens_body_r8 = nullptr;
 std::complex<float> * tens_body_c4 = nullptr;
 std::complex<double> * tens_body_c8 = nullptr;
 bool access_granted = false;
 int tens_elem_type = tens.getElementType();
 switch(tens_elem_type){
  case(talsh::REAL32): access_granted = tens.getDataAccessHost(&tens_body_r4); break;
  case(talsh::REAL64): access_granted = tens.getDataAccessHost(&tens_body_r8); break;
  case(talsh::COMPLEX32): access_granted = tens.getDataAccessHost(&tens_body_c4); break;
  case(talsh::COMPLEX64): access_granted = tens.getDataAccessHost(&tens_body_c8); break;
  default:
   std::cout << "#ERROR(exatn::runtime::node_executor_talsh): ALLREDUCE: Unknown TAL-SH data kind: "
             << tens_elem_type << std::endl;
   op.printIt();
   assert(false);
 }
 if(access_granted){
  auto mpi_data_kind = get_mpi_tensor_element_kind(tens_elem_type);
  auto communicator = *(op.getMPICommunicator().get<MPI_Comm>());
  std::size_t tens_volume = tens.getVolume();
  int chunk = std::numeric_limits<int>::max();
  for(std::size_t base = 0; base < tens_volume; base += chunk){
   int count = std::min(chunk,static_cast<int>(tens_volume-base));
   switch(tens_elem_type){
    case(talsh::REAL32):
     assert(tens_body_r4 != nullptr);
     error_code = MPI_Allreduce(MPI_IN_PLACE,(void*)(&(tens_body_r4[base])),count,mpi_data_kind,MPI_SUM,communicator);
     break;
    case(talsh::REAL64):
     assert(tens_body_r8 != nullptr);
     error_code = MPI_Allreduce(MPI_IN_PLACE,(void*)(&(tens_body_r8[base])),count,mpi_data_kind,MPI_SUM,communicator);
     break;
    case(talsh::COMPLEX32):
     assert(tens_body_c4 != nullptr);
     error_code = MPI_Allreduce(MPI_IN_PLACE,(void*)(&(tens_body_c4[base])),count,mpi_data_kind,MPI_SUM,communicator);
     break;
    case(talsh::COMPLEX64):
     assert(tens_body_c8 != nullptr);
     error_code = MPI_Allreduce(MPI_IN_PLACE,(void*)(&(tens_body_c8[base])),count,mpi_data_kind,MPI_SUM,communicator);
     break;
   }
   if(error_code != MPI_SUCCESS) break;
  }
 }else{
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): ALLREDUCE: Unable to get access to the tensor body!" << std::endl;
  op.printIt();
  assert(false);
 }
#endif
 return error_code;
}


bool TalshNodeExecutor::sync(TensorOpExecHandle op_handle,
                             int * error_code,
                             bool wait)
{
 *error_code = 0;
 bool synced = true;
 auto iter = tasks_.find(op_handle);
 if(iter != tasks_.end()){
  auto & task = *(iter->second);
  if(!task.isEmpty()){
   if(wait){
    synced = task.wait();
   }else{
    int sts;
    synced = task.test(&sts);
    if(synced && sts == TALSH_TASK_ERROR) *error_code = TALSH_TASK_ERROR;
   }
   if(synced && *error_code == 0) cacheMovedTensors(task);
  }
  if(synced) tasks_.erase(iter);
 }
 return synced;
}


bool TalshNodeExecutor::sync(bool wait)
{
 bool synced = true;

 for(auto & task: evictions_){
  bool snc = task.second->wait();
  synced = synced && snc;
 }
 evictions_.clear();

 for(auto & task: tasks_){
  bool snc = task.second->wait();
  if(snc) cacheMovedTensors(*(task.second));
  synced = synced && snc;
 }
 tasks_.clear();

 for(auto & task: prefetches_){
  bool snc = task.second->wait();
  if(snc) cacheMovedTensors(*(task.second));
  synced = synced && snc;
 }
 prefetches_.clear();

 return synced;
}


bool TalshNodeExecutor::discard(TensorOpExecHandle op_handle)
{
 auto iter = tasks_.find(op_handle);
 if(iter != tasks_.end()){
  tasks_.erase(iter);
  return true;
 }
 return false;
}


bool TalshNodeExecutor::prefetch(const numerics::TensorOperation & op)
{
 bool prefetching = false;
 const auto opcode = op.getOpcode();
 if(opcode == TensorOpCode::CONTRACT){
  const auto num_operands = op.getNumOperands(); assert(num_operands == 3);
  talsh::Tensor * talsh_tens[3];
  for(unsigned int i = 0; i < num_operands; ++i){
   auto iter = tensors_.find(op.getTensorOperand(i)->getTensorHash());
   if(iter != tensors_.end()){
    iter->second.resetTensorShapeToReduced();
    talsh_tens[i] = iter->second.talsh_tensor.get(); assert(talsh_tens[i] != nullptr);
   }else{
    return prefetching;
   }
  }
  int dev_kind;
  int opt_exec_device = talsh::determineOptimalDevice(*(talsh_tens[0]),*(talsh_tens[1]),*(talsh_tens[2]));
  int dev_id = talshKindDevId(opt_exec_device,&dev_kind);
//  std::cout << "#DEBUG(exatn::runtime::node_executor_talsh): PREFETCH: TAL-SH tensors: "
//            << talsh_tens[0] << " " << talsh_tens[1] << " " << talsh_tens[2] << std::endl << std::flush; //debug
  if(dev_kind != DEV_HOST){
   for(unsigned int i = 0; i < num_operands; ++i){
    bool in_use = tensorIsCurrentlyInUse(talsh_tens[i]);
    if(!in_use){
     auto task_res = prefetches_.emplace(std::make_pair(op.getTensorOperand(i)->getTensorHash(),
                                         std::make_shared<talsh::TensorTask>()));
     if(task_res.second){
      bool prefetch_started = talsh_tens[i]->sync(task_res.first->second.get(),dev_kind,dev_id);
      if(!prefetch_started){
       task_res.first->second->clean();
       prefetches_.erase(task_res.first);
      }
      prefetching = prefetching || prefetch_started;
     }else{
      std::cout << "#ERROR(exatn::runtime::node_executor_talsh): PREFETCH: Repeated prefetch corruption for tensor operand "
                << i << " in tensor operation:" << std::endl; //broken association between exatn::TensorHash and talsh::Tensor
      op.printIt();
      assert(false);
     }
    }
   }
  }
 }
 return prefetching;
}


std::shared_ptr<talsh::Tensor> TalshNodeExecutor::getLocalTensor(const numerics::Tensor & tensor,
                                  const std::vector<std::pair<DimOffset,DimExtent>> & slice_spec)
{
 const auto tensor_rank = slice_spec.size();
 std::vector<std::size_t> signature(tensor_rank);
 std::vector<int> offsets(tensor_rank);
 std::vector<int> dims(tensor_rank);
 for(unsigned int i = 0; i < tensor_rank; ++i){
  signature[i] = static_cast<std::size_t>(slice_spec[i].first);
  offsets[i] = static_cast<int>(slice_spec[i].first);
  dims[i] = static_cast<int>(slice_spec[i].second);
 }
 std::shared_ptr<talsh::Tensor> slice(nullptr);
 switch(tensor.getElementType()){
  case TensorElementType::REAL32:
   slice = std::make_shared<talsh::Tensor>(signature,dims,static_cast<float>(0.0));
   break;
  case TensorElementType::REAL64:
   slice = std::make_shared<talsh::Tensor>(signature,dims,static_cast<double>(0.0));
   break;
  case TensorElementType::COMPLEX32:
   slice = std::make_shared<talsh::Tensor>(signature,dims,std::complex<float>(0.0));
   break;
  case TensorElementType::COMPLEX64:
   slice = std::make_shared<talsh::Tensor>(signature,dims,std::complex<double>(0.0));
   break;
  default:
   std::cout << "#ERROR(exatn::runtime::TalshNodeExecutor::getLocalTensor): Invalid tensor element type!" << std::endl;
   std::abort();
 }
 auto tens_pos = tensors_.find(tensor.getTensorHash());
 if(tens_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::TalshNodeExecutor::getLocalTensor): Tensor not found: " << std::endl;
  tensor.printIt();
  std::abort();
 }
 tens_pos->second.resetTensorShapeToFull();
 auto & talsh_tensor = *(tens_pos->second.talsh_tensor);
 auto error_code = talsh_tensor.extractSlice(nullptr,*slice,offsets); assert(error_code == TALSH_SUCCESS);
 return slice;
}


bool TalshNodeExecutor::finishPrefetching(const numerics::TensorOperation & op)
{
 bool synced = true;
 //Test completion of active evictions:
 auto eviction = evictions_.begin();
 while(eviction != evictions_.end()){
  int sts;
  bool snc = eviction->second->test(&sts);
  if(snc) eviction = evictions_.erase(eviction);
 }
 //Test completion of active prefetches:
 auto prefetch = prefetches_.begin();
 while(prefetch != prefetches_.end()){
  int sts;
  bool snc = prefetch->second->test(&sts);
  if(snc){
   cacheMovedTensors(*(prefetch->second));
   prefetch = prefetches_.erase(prefetch);
  }
 }
 //Finish tensor operand prefetching for the given tensor operation:
 const auto num_operands = op.getNumOperands();
 for(unsigned int oprnd = 0; oprnd < num_operands; ++oprnd){
  const auto tens_hash = op.getTensorOperand(oprnd)->getTensorHash();
  auto iter = prefetches_.find(tens_hash);
  if(iter != prefetches_.end()){
   bool snc = iter->second->wait();
   if(snc){
    cacheMovedTensors(*(iter->second));
    prefetches_.erase(iter);
   }
   synced = synced && snc;
  }
 }
 return synced;
}


void TalshNodeExecutor::cacheMovedTensors(talsh::TensorTask & talsh_task)
{
 if(!(talsh_task.isEmpty())){
  int dev_kind;
  int dev_id = talsh_task.getExecutionDevice(&dev_kind);
  if(dev_kind != DEV_HOST){
   const int device = talshFlatDevId(dev_kind,dev_id);
   const auto num_operands = talsh_task.getNumTensorArguments();
   const auto coherence = talsh_task.getTensorArgumentCoherence();
   if(coherence >= 0){
    for(unsigned int oprnd = 0; oprnd < num_operands; ++oprnd){
     unsigned int arg_coherence = argument_coherence_get_value(coherence,num_operands,oprnd);
     if(arg_coherence == COPY_M || arg_coherence == COPY_K){ //move or copy of tensor body image
      auto res = accel_cache_[device].emplace(std::make_pair(const_cast<talsh::Tensor*>(talsh_task.getTensorArgument(oprnd)),
                                                             CachedAttr{exatn::Timer::timeInSecHR()}));
     }
    }
   }
  }
 }
 return;
}


bool TalshNodeExecutor::evictMovedTensors(int device_id, std::size_t required_space)
{
 bool evicting = false, single_device = false;
 int dev_begin = 0, dev_end = (DEV_MAX - 1);
 if(device_id != DEV_DEFAULT){
  assert(device_id >= 0 && device_id < DEV_MAX);
  dev_begin = device_id; dev_end = device_id;
  single_device = true;
 }
 std::size_t freed_bytes = 0;
 bool still_freeing = true;
 while(still_freeing){
  bool cache_empty = true;
  for(int dev = dev_begin; dev <= dev_end; ++dev){
   auto iter = accel_cache_[dev].begin();
   while(iter != accel_cache_[dev].end()){
    if(!tensorIsCurrentlyInUse(iter->first)){
     cache_empty = false;
     int data_kind_size;
     auto valid = talshValidDataKind(iter->first->getElementType(),&data_kind_size);
     assert(valid == YEP);
     std::size_t talsh_tens_size = iter->first->getVolume() * data_kind_size;
     auto task_res = evictions_.emplace(std::make_pair(iter->first,
                                                       std::make_shared<talsh::TensorTask>()));
     if(task_res.second){
      bool synced = iter->first->sync(task_res.first->second.get(),DEV_HOST,0,nullptr,!single_device); //initiate a move of the tensor body image back to Host
      iter = accel_cache_[dev].erase(iter);
      freed_bytes += talsh_tens_size;
      iter = accel_cache_[dev].end();
      evicting = true;
     }
    }else{
     ++iter;
    }
   }
   still_freeing = ((!cache_empty) || (dev < dev_end)) && ((required_space == 0) || (freed_bytes < required_space));
   if(!still_freeing) break;
  }
 }
 return evicting;
}


bool TalshNodeExecutor::tensorIsCurrentlyInUse(const talsh::Tensor * talsh_tens) const
{
 for(const auto & task: evictions_){
  const auto num_task_args = task.second->getNumTensorArguments();
  for(unsigned int i = 0; i < num_task_args; ++i){
   if(task.second->getTensorArgument(i) == talsh_tens) return true;
  }
 }
 for(const auto & task: tasks_){
  const auto num_task_args = task.second->getNumTensorArguments();
  for(unsigned int i = 0; i < num_task_args; ++i){
   if(task.second->getTensorArgument(i) == talsh_tens) return true;
  }
 }
 for(const auto & task: prefetches_){
  const auto num_task_args = task.second->getNumTensorArguments();
  for(unsigned int i = 0; i < num_task_args; ++i){
   if(task.second->getTensorArgument(i) == talsh_tens) return true;
  }
 }
 return false;
}

} //namespace runtime
} //namespace exatn
