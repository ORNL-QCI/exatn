#include "node_executor_talsh.hpp"

namespace exatn {
namespace runtime {

TensorOpExecHandle TalshNodeExecutor::execute(numerics::TensorOpCreate & op)
{
 //`Implement
 return 0;
}


TensorOpExecHandle TalshNodeExecutor::execute(numerics::TensorOpDestroy & op)
{
 //`Implement
 return 0;
}


TensorOpExecHandle TalshNodeExecutor::execute(numerics::TensorOpTransform & op)
{
 //`Implement
 return 0;
}


TensorOpExecHandle TalshNodeExecutor::execute(numerics::TensorOpAdd & op)
{
 //`Implement
 return 0;
}


TensorOpExecHandle TalshNodeExecutor::execute(numerics::TensorOpContract & op)
{
 //`Implement
 return 0;
}


bool TalshNodeExecutor::sync(TensorOpExecHandle op_handle,
                             int * error_code,
                             bool wait)
{
 //`Implement
 return false;
}

} //namespace runtime
} //namespace exatn
