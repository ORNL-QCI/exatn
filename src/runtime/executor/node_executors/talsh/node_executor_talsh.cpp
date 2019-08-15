#include "node_executor_talsh.hpp"

namespace exatn {
namespace runtime {

NodeExecHandleType TalshNodeExecutor::execute(numerics::TensorOpCreate & op)
{
 //`Implement
 return 0;
}


NodeExecHandleType TalshNodeExecutor::execute(numerics::TensorOpDestroy & op)
{
 //`Implement
 return 0;
}


NodeExecHandleType TalshNodeExecutor::execute(numerics::TensorOpTransform & op)
{
 //`Implement
 return 0;
}


NodeExecHandleType TalshNodeExecutor::execute(numerics::TensorOpAdd & op)
{
 //`Implement
 return 0;
}


NodeExecHandleType TalshNodeExecutor::execute(numerics::TensorOpContract & op)
{
 //`Implement
 return 0;
}


bool TalshNodeExecutor::sync(NodeExecHandleType op_handle, bool wait)
{
 //`Implement
 return false;
}

} //namespace runtime
} //namespace exatn
