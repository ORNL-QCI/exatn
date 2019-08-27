#include "node_executor_exatensor.hpp"

namespace exatn {
namespace runtime {

TensorOpExecHandle ExatensorNodeExecutor::execute(numerics::TensorOpCreate & op)
{
 //`Implement
 return 0;
}


TensorOpExecHandle ExatensorNodeExecutor::execute(numerics::TensorOpDestroy & op)
{
 //`Implement
 return 0;
}


TensorOpExecHandle ExatensorNodeExecutor::execute(numerics::TensorOpTransform & op)
{
 //`Implement
 return 0;
}


TensorOpExecHandle ExatensorNodeExecutor::execute(numerics::TensorOpAdd & op)
{
 //`Implement
 return 0;
}


TensorOpExecHandle ExatensorNodeExecutor::execute(numerics::TensorOpContract & op)
{
 //`Implement
 return 0;
}


bool ExatensorNodeExecutor::sync(TensorOpExecHandle op_handle,
                                 int * error_code,
                                 bool wait)
{
 //`Implement
 return false;
}

} //namespace runtime
} //namespace exatn
