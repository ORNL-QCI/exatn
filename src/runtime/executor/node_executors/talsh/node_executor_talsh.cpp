/** ExaTN:: Tensor Runtime: Tensor graph node executor: Talsh
REVISION: 2020/06/05

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
 talsh_init_lock.lock();
 if(!talsh_initialized_){
  std::size_t host_mem_buffer_size = DEFAULT_MEM_BUFFER_SIZE;
  int64_t provided_buf_size = 0;
  if(parameters.getParameter("host_memory_buffer_size",&provided_buf_size))
   host_mem_buffer_size = provided_buf_size;
  auto error_code = talsh::initialize(&host_mem_buffer_size);
  if(error_code == TALSH_SUCCESS){
   talsh_host_mem_buffer_size_.store(host_mem_buffer_size);
   std::cout << "#DEBUG(exatn::runtime::TalshNodeExecutor): TAL-SH initialized with Host buffer size of " <<
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
 talsh_init_lock.lock();
 --talsh_node_exec_count_;
 if(talsh_initialized_ && talsh_node_exec_count_ == 0){
  tasks_.clear();
  tensors_.clear();
  talsh::printStatistics();
  auto error_code = talsh::shutdown();
  if(error_code == TALSH_SUCCESS){
   std::cout << "#DEBUG(exatn::runtime::TalshNodeExecutor): TAL-SH shut down" << std::endl << std::flush;
   talsh_initialized_ = false;
  }else{
   std::cerr << "#FATAL(exatn::runtime::TalshNodeExecutor): Unable to shut down TAL-SH!" << std::endl;
   assert(false);
  }
 }
 talsh_init_lock.unlock();
}


int TalshNodeExecutor::execute(numerics::TensorOpCreate & op,
                               TensorOpExecHandle * exec_handle)
{
 assert(op.isSet());
 const auto & tensor = *(op.getTensorOperand(0));
 const auto tensor_rank = tensor.getRank();
 const auto tensor_hash = tensor.getTensorHash();
 const auto & dim_extents = tensor.getDimExtents();
 std::vector<int> extents(tensor_rank);
 for(int i = 0; i < tensor_rank; ++i) extents[i] = static_cast<int>(dim_extents[i]);
 auto data_kind = get_talsh_tensor_element_kind(op.getTensorElementType());
 auto res = tensors_.emplace(std::make_pair(tensor_hash,
                             std::make_shared<talsh::Tensor>(extents,data_kind,talsh_tens_no_init)));
 if(!res.second){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): CREATE: Attempt to create the same tensor twice: " << std::endl;
  tensor.printIt();
  assert(false);
 }else{
  //std::cout << "#DEBUG(exatn::runtime::node_executor_talsh): New tensor " << tensor.getName()
  //          << " emplaced with hash " << tensor_hash << std::endl;
 }
 *exec_handle = op.getId();
 return 0;
}


int TalshNodeExecutor::execute(numerics::TensorOpDestroy & op,
                               TensorOpExecHandle * exec_handle)
{
 assert(op.isSet());
 const auto & tensor = *(op.getTensorOperand(0));
 const auto tensor_hash = tensor.getTensorHash();
 auto num_deleted = tensors_.erase(tensor_hash);
 if(num_deleted != 1){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): DESTROY: Attempt to destroy non-existing tensor: " << std::endl;
  tensor.printIt();
  assert(false);
 }else{
  //std::cout << "#DEBUG(exatn::runtime::node_executor_talsh): Tensor " << tensor.getName()
  //          << " erased with hash " << tensor_hash << std::endl;
 }
 *exec_handle = op.getId();
 return 0;
}


int TalshNodeExecutor::execute(numerics::TensorOpTransform & op,
                               TensorOpExecHandle * exec_handle)
{
 assert(op.isSet());
 const auto & tensor = *(op.getTensorOperand(0));
 const auto tensor_hash = tensor.getTensorHash();
 auto tens_pos = tensors_.find(tensor_hash);
 if(tens_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): TRANSFORM: Tensor operand 0 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 auto & tens = *(tens_pos->second);
 int error_code = op.apply(tens); //synchronous user-defined Host operation
 *exec_handle = op.getId();
 return error_code;
}


int TalshNodeExecutor::execute(numerics::TensorOpSlice & op,
                               TensorOpExecHandle * exec_handle)
{
 assert(op.isSet());
 const auto & tensor0 = *(op.getTensorOperand(0));
 const auto tensor0_hash = tensor0.getTensorHash();
 auto tens0_pos = tensors_.find(tensor0_hash);
 if(tens0_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): SLICE: Tensor operand 0 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 auto & tens0 = *(tens0_pos->second);

 const auto & tensor1 = *(op.getTensorOperand(1));
 const auto tensor1_hash = tensor1.getTensorHash();
 auto tens1_pos = tensors_.find(tensor1_hash);
 if(tens1_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): SLICE: Tensor operand 1 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 auto & tens1 = *(tens1_pos->second);

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
 const auto & tensor0 = *(op.getTensorOperand(0));
 const auto tensor0_hash = tensor0.getTensorHash();
 auto tens0_pos = tensors_.find(tensor0_hash);
 if(tens0_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): INSERT: Tensor operand 0 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 auto & tens0 = *(tens0_pos->second);

 const auto & tensor1 = *(op.getTensorOperand(1));
 const auto tensor1_hash = tensor1.getTensorHash();
 auto tens1_pos = tensors_.find(tensor1_hash);
 if(tens1_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): INSERT: Tensor operand 1 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 auto & tens1 = *(tens1_pos->second);

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
 const auto & tensor0 = *(op.getTensorOperand(0));
 const auto tensor0_hash = tensor0.getTensorHash();
 auto tens0_pos = tensors_.find(tensor0_hash);
 if(tens0_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): ADD: Tensor operand 0 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 auto & tens0 = *(tens0_pos->second);

 const auto & tensor1 = *(op.getTensorOperand(1));
 const auto tensor1_hash = tensor1.getTensorHash();
 auto tens1_pos = tensors_.find(tensor1_hash);
 if(tens1_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): ADD: Tensor operand 1 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 auto & tens1 = *(tens1_pos->second);

 *exec_handle = op.getId();
 auto task_res = tasks_.emplace(std::make_pair(*exec_handle,
                                std::make_shared<talsh::TensorTask>()));
 if(!task_res.second){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): ADD: Attempt to execute the same operation twice: " << std::endl;
  op.printIt();
  assert(false);
 }

 auto error_code = tens0.accumulate((task_res.first)->second.get(),
                                    op.getIndexPattern(),
                                    tens1,
                                    DEV_DEFAULT,DEV_DEFAULT,
                                    op.getScalar(0));
 return error_code;
}


int TalshNodeExecutor::execute(numerics::TensorOpContract & op,
                               TensorOpExecHandle * exec_handle)
{
 assert(op.isSet());
 const auto & tensor0 = *(op.getTensorOperand(0));
 const auto tensor0_hash = tensor0.getTensorHash();
 auto tens0_pos = tensors_.find(tensor0_hash);
 if(tens0_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): CONTRACT: Tensor operand 0 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 auto & tens0 = *(tens0_pos->second);

 const auto & tensor1 = *(op.getTensorOperand(1));
 const auto tensor1_hash = tensor1.getTensorHash();
 auto tens1_pos = tensors_.find(tensor1_hash);
 if(tens1_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): CONTRACT: Tensor operand 1 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 auto & tens1 = *(tens1_pos->second);

 const auto & tensor2 = *(op.getTensorOperand(2));
 const auto tensor2_hash = tensor2.getTensorHash();
 auto tens2_pos = tensors_.find(tensor2_hash);
 if(tens2_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): CONTRACT: Tensor operand 2 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 auto & tens2 = *(tens2_pos->second);

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
                                            op.getIndexPattern(),
                                            tens1,tens2,
                                            DEV_DEFAULT,DEV_DEFAULT,
                                            op.getScalar(0));
 return error_code;
}


int TalshNodeExecutor::execute(numerics::TensorOpDecomposeSVD3 & op,
                               TensorOpExecHandle * exec_handle)
{
 assert(op.isSet());
 const auto & tensor0 = *(op.getTensorOperand(0));
 const auto tensor0_hash = tensor0.getTensorHash();
 auto tens0_pos = tensors_.find(tensor0_hash);
 if(tens0_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): DECOMPOSE_SVD3: Tensor operand 0 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 auto & tens0 = *(tens0_pos->second);

 const auto & tensor1 = *(op.getTensorOperand(1));
 const auto tensor1_hash = tensor1.getTensorHash();
 auto tens1_pos = tensors_.find(tensor1_hash);
 if(tens1_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): DECOMPOSE_SVD3: Tensor operand 1 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 auto & tens1 = *(tens1_pos->second);

 const auto & tensor2 = *(op.getTensorOperand(2));
 const auto tensor2_hash = tensor2.getTensorHash();
 auto tens2_pos = tensors_.find(tensor2_hash);
 if(tens2_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): DECOMPOSE_SVD3: Tensor operand 2 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 auto & tens2 = *(tens2_pos->second);

 const auto & tensor3 = *(op.getTensorOperand(3));
 const auto tensor3_hash = tensor3.getTensorHash();
 auto tens3_pos = tensors_.find(tensor3_hash);
 if(tens3_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): DECOMPOSE_SVD3: Tensor operand 3 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 auto & tens3 = *(tens3_pos->second);

 *exec_handle = op.getId();
 auto task_res = tasks_.emplace(std::make_pair(*exec_handle,
                                std::make_shared<talsh::TensorTask>()));
 if(!task_res.second){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): DECOMPOSE_SVD3: Attempt to execute the same operation twice: " << std::endl;
  op.printIt();
  assert(false);
 }

 auto error_code = tens3.decomposeSVD((task_res.first)->second.get(),
                                      op.getIndexPattern(),
                                      tens0,tens1,tens2,
                                      DEV_HOST,0);
 return error_code;
}


int TalshNodeExecutor::execute(numerics::TensorOpDecomposeSVD2 & op,
                               TensorOpExecHandle * exec_handle)
{
 assert(op.isSet());
 const auto & tensor0 = *(op.getTensorOperand(0));
 const auto tensor0_hash = tensor0.getTensorHash();
 auto tens0_pos = tensors_.find(tensor0_hash);
 if(tens0_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): DECOMPOSE_SVD2: Tensor operand 0 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 auto & tens0 = *(tens0_pos->second);

 const auto & tensor1 = *(op.getTensorOperand(1));
 const auto tensor1_hash = tensor1.getTensorHash();
 auto tens1_pos = tensors_.find(tensor1_hash);
 if(tens1_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): DECOMPOSE_SVD2: Tensor operand 1 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 auto & tens1 = *(tens1_pos->second);

 const auto & tensor2 = *(op.getTensorOperand(2));
 const auto tensor2_hash = tensor2.getTensorHash();
 auto tens2_pos = tensors_.find(tensor2_hash);
 if(tens2_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): DECOMPOSE_SVD2: Tensor operand 2 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 auto & tens2 = *(tens2_pos->second);

 *exec_handle = op.getId();
 auto task_res = tasks_.emplace(std::make_pair(*exec_handle,
                                std::make_shared<talsh::TensorTask>()));
 if(!task_res.second){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): DECOMPOSE_SVD2: Attempt to execute the same operation twice: " << std::endl;
  op.printIt();
  assert(false);
 }

 auto error_code = tens2.decomposeSVDLR((task_res.first)->second.get(),
                                        op.getIndexPattern(),
                                        tens0,tens1,
                                        DEV_HOST,0);
 return error_code;
}


int TalshNodeExecutor::execute(numerics::TensorOpOrthogonalizeSVD & op,
                               TensorOpExecHandle * exec_handle)
{
 assert(op.isSet());
 const auto & tensor0 = *(op.getTensorOperand(0));
 const auto tensor0_hash = tensor0.getTensorHash();
 auto tens0_pos = tensors_.find(tensor0_hash);
 if(tens0_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): ORTHOGONALIZE_SVD: Tensor operand 0 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 auto & tens0 = *(tens0_pos->second);

 *exec_handle = op.getId();
 auto task_res = tasks_.emplace(std::make_pair(*exec_handle,
                                std::make_shared<talsh::TensorTask>()));
 if(!task_res.second){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): ORTHOGONALIZE_SVD: Attempt to execute the same operation twice: " << std::endl;
  op.printIt();
  assert(false);
 }

 auto error_code = tens0.orthogonalizeSVD((task_res.first)->second.get(),
                                          op.getIndexPattern(),
                                          DEV_HOST,0);
 return error_code;
}


int TalshNodeExecutor::execute(numerics::TensorOpOrthogonalizeMGS & op,
                               TensorOpExecHandle * exec_handle)
{
 assert(op.isSet());
 const auto & tensor0 = *(op.getTensorOperand(0));
 const auto tensor0_hash = tensor0.getTensorHash();
 auto tens0_pos = tensors_.find(tensor0_hash);
 if(tens0_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): ORTHOGONALIZE_MGS: Tensor operand 0 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 auto & tens0 = *(tens0_pos->second);

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
                                          op.getIndexPattern(),
                                          DEV_HOST,0);
 */
 return error_code;
}


int TalshNodeExecutor::execute(numerics::TensorOpBroadcast & op,
                               TensorOpExecHandle * exec_handle)
{
 assert(op.isSet());
 const auto & tensor = *(op.getTensorOperand(0));
 const auto tensor_hash = tensor.getTensorHash();
 auto tens_pos = tensors_.find(tensor_hash);
 if(tens_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): BROADCAST: Tensor operand 0 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 auto & tens = *(tens_pos->second);

 *exec_handle = op.getId();

 int error_code = 0;
#ifdef MPI_ENABLED
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
 const auto & tensor = *(op.getTensorOperand(0));
 const auto tensor_hash = tensor.getTensorHash();
 auto tens_pos = tensors_.find(tensor_hash);
 if(tens_pos == tensors_.end()){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): ALLREDUCE: Tensor operand 0 not found: " << std::endl;
  op.printIt();
  assert(false);
 }
 auto & tens = *(tens_pos->second);

 *exec_handle = op.getId();

 int error_code = 0;
#ifdef MPI_ENABLED
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
  }
  if(synced) tasks_.erase(iter);
 }
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
 auto & talsh_tensor = *(tens_pos->second);
 auto error_code = talsh_tensor.extractSlice(nullptr,*slice,offsets); assert(error_code == TALSH_SUCCESS);
 return slice;
}

} //namespace runtime
} //namespace exatn
