/** ExaTN::Numerics: Tensor operation: Fetches remote tensor data
REVISION: 2021/07/15

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "exatn_service.hpp"

#include "tensor_op_fetch.hpp"

#include "tensor_node_executor.hpp"

namespace exatn{

namespace numerics{

TensorOpFetch::TensorOpFetch():
 TensorOperation(TensorOpCode::FETCH,1,0,1,{0}),
 remote_rank_(-1), message_tag_(0)
{
}

bool TensorOpFetch::isSet() const
{
 return (this->getNumOperandsSet() == this->getNumOperands());
}

int TensorOpFetch::accept(runtime::TensorNodeExecutor & node_executor,
                          runtime::TensorOpExecHandle * exec_handle)
{
 return node_executor.execute(*this,exec_handle);
}

std::unique_ptr<TensorOperation> TensorOpFetch::createNew()
{
 return std::unique_ptr<TensorOperation>(new TensorOpFetch());
}

bool TensorOpFetch::resetMPICommunicator(const MPICommProxy & intra_comm)
{
 intra_comm_ = intra_comm;
 return true;
}

const MPICommProxy & TensorOpFetch::getMPICommunicator() const
{
 return intra_comm_;
}

bool TensorOpFetch::resetRemoteProcessRank(unsigned int rank)
{
 remote_rank_ = static_cast<int>(rank);
 return true;
}

int TensorOpFetch::getRemoteProcessRank() const
{
 return remote_rank_;
}

bool TensorOpFetch::resetMessageTag(int tag)
{
 message_tag_ = tag;
 return true;
}

int TensorOpFetch::getMessageTag() const
{
 return message_tag_;
}

std::size_t TensorOpFetch::decompose(const TensorMapper & tensor_mapper)
{
 assert(false);
 //`Implement
 return 0;
}

} //namespace numerics

} //namespace exatn
