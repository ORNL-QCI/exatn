/** ExaTN::Numerics: Tensor operation: Broadcasts a tensor
REVISION: 2021/07/14

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "exatn_service.hpp"

#include "tensor_op_broadcast.hpp"

#include "tensor_node_executor.hpp"

namespace exatn{

namespace numerics{

TensorOpBroadcast::TensorOpBroadcast():
 TensorOperation(TensorOpCode::BROADCAST,1,0,1,{0}),
 root_rank_(0)
{
}

bool TensorOpBroadcast::isSet() const
{
 return (this->getNumOperandsSet() == this->getNumOperands());
}

int TensorOpBroadcast::accept(runtime::TensorNodeExecutor & node_executor,
                              runtime::TensorOpExecHandle * exec_handle)
{
 return node_executor.execute(*this,exec_handle);
}

std::unique_ptr<TensorOperation> TensorOpBroadcast::createNew()
{
 return std::unique_ptr<TensorOperation>(new TensorOpBroadcast());
}

bool TensorOpBroadcast::resetMPICommunicator(const MPICommProxy & intra_comm)
{
 intra_comm_ = intra_comm;
 return true;
}

const MPICommProxy & TensorOpBroadcast::getMPICommunicator() const
{
 return intra_comm_;
}

bool TensorOpBroadcast::resetRootRank(unsigned int rank)
{
 root_rank_ = static_cast<int>(rank);
 return true;
}

int TensorOpBroadcast::getRootRank() const
{
 return root_rank_;
}

std::size_t TensorOpBroadcast::decompose(std::function<bool (const Tensor &)> tensor_exists_locally)
{
 assert(false);
 //`Implement
 return 0;
}

} //namespace numerics

} //namespace exatn
