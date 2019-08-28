#include "node_executor_talsh.hpp"

#include <cassert>

namespace exatn {
namespace runtime {

bool TalshNodeExecutor::talsh_initialized_ = 0;

TensorOpExecHandle TalshNodeExecutor::execute(numerics::TensorOpCreate & op)
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
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): Attempt to create the same tensor twice for tensor: " << std::endl;
  tensor.printIt();
  assert(false);
 }
 return op.getId();
}


TensorOpExecHandle TalshNodeExecutor::execute(numerics::TensorOpDestroy & op)
{
 assert(op.isSet());
 const auto & tensor = *(op.getTensorOperand(0));
 const auto tensor_hash = tensor.getTensorHash();
 auto num_deleted = tensors_.erase(tensor_hash);
 if(num_deleted != 1){
  std::cout << "#ERROR(exatn::runtime::node_executor_talsh): Attempt to destroy non-existing tensor for tensor: " << std::endl;
  tensor.printIt();
  assert(false);
 }
 return op.getId();
}


TensorOpExecHandle TalshNodeExecutor::execute(numerics::TensorOpTransform & op)
{
 assert(op.isSet());
 //`Implement
 return op.getId();
}


TensorOpExecHandle TalshNodeExecutor::execute(numerics::TensorOpAdd & op)
{
 assert(op.isSet());
 //`Implement
 return op.getId();
}


TensorOpExecHandle TalshNodeExecutor::execute(numerics::TensorOpContract & op)
{
 assert(op.isSet());
 //`Implement
 return op.getId();
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
  assert(!task.isEmpty());
  if(wait){
   synced = task.wait();
  }else{
   int sts;
   synced = task.test(&sts);
  }
  if(synced) tasks_.erase(iter);
 }
 return synced;
}

} //namespace runtime
} //namespace exatn
