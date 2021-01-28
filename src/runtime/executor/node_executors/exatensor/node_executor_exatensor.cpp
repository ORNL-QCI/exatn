/** ExaTN:: Tensor Runtime: Tensor graph node executor: Exatensor
REVISION: 2021/01/28

Copyright (C) 2018-2021 Dmitry Lyakh, Tiffany Mintz, Alex McCaskey
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle)
**/

#include "node_executor_exatensor.hpp"

#include "errors.hpp"

namespace exatn {
namespace runtime {

void ExatensorNodeExecutor::initialize(const ParamConf & parameters)
{
 return;
}


void ExatensorNodeExecutor::activateFastMath()
{
 //`Finish
 return;
}


std::size_t ExatensorNodeExecutor::getMemoryBufferSize() const
{
 std::size_t buf_size = 0;
 while(buf_size == 0) buf_size = exatensor_host_mem_buffer_size_.load();
 return buf_size;
}


double ExatensorNodeExecutor::getTotalFlopCount() const
{
 //`Implement
 return 0.0;
}


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


int ExatensorNodeExecutor::execute(numerics::TensorOpSlice & op,
                                   TensorOpExecHandle * exec_handle)
{
 //`Implement
 return 0;
}


int ExatensorNodeExecutor::execute(numerics::TensorOpInsert & op,
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


int ExatensorNodeExecutor::execute(numerics::TensorOpDecomposeSVD3 & op,
                                   TensorOpExecHandle * exec_handle)
{
 //`Implement
 return 0;
}


int ExatensorNodeExecutor::execute(numerics::TensorOpDecomposeSVD2 & op,
                                   TensorOpExecHandle * exec_handle)
{
 //`Implement
 return 0;
}


int ExatensorNodeExecutor::execute(numerics::TensorOpOrthogonalizeSVD & op,
                                   TensorOpExecHandle * exec_handle)
{
 //`Implement
 return 0;
}


int ExatensorNodeExecutor::execute(numerics::TensorOpOrthogonalizeMGS & op,
                                   TensorOpExecHandle * exec_handle)
{
 //`Implement
 return 0;
}


int ExatensorNodeExecutor::execute(numerics::TensorOpBroadcast & op,
                                   TensorOpExecHandle * exec_handle)
{
 //`Implement
 return 0;
}


int ExatensorNodeExecutor::execute(numerics::TensorOpAllreduce & op,
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


bool ExatensorNodeExecutor::sync()
{
 //`Implement
 return false;
}


bool ExatensorNodeExecutor::discard(TensorOpExecHandle op_handle)
{
 //`Implement
 return false;
}


bool ExatensorNodeExecutor::prefetch(const numerics::TensorOperation & op)
{
 //`Implement
 return false;
}


std::shared_ptr<talsh::Tensor> ExatensorNodeExecutor::getLocalTensor(const numerics::Tensor & tensor,
                                      const std::vector<std::pair<DimOffset,DimExtent>> & slice_spec)
{
 //`Implement
 return std::make_shared<talsh::Tensor>(std::vector<int>{},0.0);
}

} //namespace runtime
} //namespace exatn
