/** ExaTN::Numerics: Tensor operation: All-reduces a tensor
REVISION: 2021/07/14

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "exatn_service.hpp"

#include "tensor_op_allreduce.hpp"

#include "tensor_node_executor.hpp"

namespace exatn{

namespace numerics{

TensorOpAllreduce::TensorOpAllreduce():
 TensorOperation(TensorOpCode::ALLREDUCE,1,0,1,{0})
{
}

bool TensorOpAllreduce::isSet() const
{
 return (this->getNumOperandsSet() == this->getNumOperands());
}

int TensorOpAllreduce::accept(runtime::TensorNodeExecutor & node_executor,
                              runtime::TensorOpExecHandle * exec_handle)
{
 return node_executor.execute(*this,exec_handle);
}

std::unique_ptr<TensorOperation> TensorOpAllreduce::createNew()
{
 return std::unique_ptr<TensorOperation>(new TensorOpAllreduce());
}

bool TensorOpAllreduce::resetMPICommunicator(const MPICommProxy & intra_comm)
{
 intra_comm_ = intra_comm;
 return true;
}

const MPICommProxy & TensorOpAllreduce::getMPICommunicator() const
{
 return intra_comm_;
}

std::size_t TensorOpAllreduce::decompose(std::function<bool (const Tensor &)> tensor_exists_locally)
{
 assert(false);
 //`Implement
 return 0;
}

} //namespace numerics

} //namespace exatn
