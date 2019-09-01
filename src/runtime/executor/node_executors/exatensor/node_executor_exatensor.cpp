#include "node_executor_exatensor.hpp"

namespace exatn {
namespace runtime {

int ExatensorNodeExecutor::execute(numerics::TensorOpCreate & op,
                                   TensorOpExecHandle * exec_handle)
{
 //`Implement
 return 0;
}


int ExatensorNodeExecutor::execute(numerics::TensorOpDestroy & op,
                                   TensorOpExecHandle * exec_handle)
{
 //`Implement
 return 0;
}


int ExatensorNodeExecutor::execute(numerics::TensorOpTransform & op,
                                   TensorOpExecHandle * exec_handle)
{
 //`Implement
 return 0;
}


int ExatensorNodeExecutor::execute(numerics::TensorOpAdd & op,
                                   TensorOpExecHandle * exec_handle)
{
 //`Implement
 return 0;
}


int ExatensorNodeExecutor::execute(numerics::TensorOpContract & op,
                                   TensorOpExecHandle * exec_handle)
{
 //`Implement
 return 0;
}


bool ExatensorNodeExecutor::sync(TensorOpExecHandle op_handle,
                                 int * error_code,
                                 bool wait)
{
 *error_code = 0;
 //`Implement
 return false;
}

} //namespace runtime
} //namespace exatn
