#include "node_executor_talsh.hpp"

#include <mutex>
#include <cassert>

namespace exatn {
namespace runtime {

bool TalshNodeExecutor::talsh_initialized_ = false;
int TalshNodeExecutor::talsh_node_exec_count_ = 0;

std::mutex talsh_init_lock;


void TalshNodeExecutor::initialize()
{
 talsh_init_lock.lock();
 if(!talsh_initialized_){
  std::size_t host_buffer_size = 1024*1024*1024; //`Get max Host memory from OS
  auto error_code = talsh::initialize(&host_buffer_size);
  if(error_code == TALSH_SUCCESS){
   //std::cout << "#DEBUG(exatn::runtime::TalshNodeExecutor): TAL-SH initialized with Host buffer size of " <<
    //host_buffer_size << " Bytes" << std::endl << std::flush;
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


TalshNodeExecutor::~TalshNodeExecutor()
{
 talsh_init_lock.lock();
 --talsh_node_exec_count_;
 if(talsh_initialized_ && talsh_node_exec_count_ == 0){
  auto error_code = talsh::shutdown();
  if(error_code == TALSH_SUCCESS){
   //std::cout << "#DEBUG(exatn::runtime::TalshNodeExecutor): TAL-SH shut down" << std::endl << std::flush;
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
 }
 *exec_handle = op.getId();
 return 0;
}


int TalshNodeExecutor::execute(numerics::TensorOpTransform & op,
                               TensorOpExecHandle * exec_handle)
{
 assert(op.isSet());
 //`Implement
 *exec_handle = op.getId();
 return 0;
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
                                    DEV_HOST,0,
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

 auto error_code = tens0.contractAccumulate((task_res.first)->second.get(),
                                            op.getIndexPattern(),
                                            tens1,tens2,
                                            DEV_HOST,0,
                                            op.getScalar(0));
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

} //namespace runtime
} //namespace exatn
