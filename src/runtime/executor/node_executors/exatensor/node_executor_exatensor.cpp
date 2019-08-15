#include "node_executor_exatensor.hpp"

namespace exatn {
namespace runtime {

NodeExecHandleType ExatensorNodeExecutor::execute(numerics::TensorOpCreate & op)
{
 //`Implement
 return 0;
}


NodeExecHandleType ExatensorNodeExecutor::execute(numerics::TensorOpDestroy & op)
{
 //`Implement
 return 0;
}


NodeExecHandleType ExatensorNodeExecutor::execute(numerics::TensorOpTransform & op)
{
 //`Implement
 return 0;
}


NodeExecHandleType ExatensorNodeExecutor::execute(numerics::TensorOpAdd & op)
{
 //`Implement
 return 0;
}


NodeExecHandleType ExatensorNodeExecutor::execute(numerics::TensorOpContract & op)
{
 //`Implement
 return 0;
}


bool ExatensorNodeExecutor::sync(NodeExecHandleType op_handle, bool wait)
{
 //`Implement
 return false;
}

} //namespace runtime
} //namespace exatn
