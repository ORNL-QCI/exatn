#include "node_executor_talsh.hpp"

#include <cassert>

namespace exatn {
namespace runtime {

TensorOpExecHandle TalshNodeExecutor::execute(numerics::TensorOpCreate & op)
{
 assert(op.isSet());
 const auto & tensor = *(op.getTensorOperand(0));
 //`Implement
 return op.getId();
}


TensorOpExecHandle TalshNodeExecutor::execute(numerics::TensorOpDestroy & op)
{
 assert(op.isSet());
 //`Implement
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
