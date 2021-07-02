/** ExaTN::Numerics: Tensor operation: Uploads remote tensor data
REVISION: 2021/07/02

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "exatn_service.hpp"

#include "tensor_op_upload.hpp"

#include "tensor_node_executor.hpp"

namespace exatn{

namespace numerics{

TensorOpUpload::TensorOpUpload():
 TensorOperation(TensorOpCode::UPLOAD,1,0,1,{0}),
 remote_rank_(-1), message_tag_(0)
{
}

bool TensorOpUpload::isSet() const
{
 return (this->getNumOperandsSet() == this->getNumOperands());
}

int TensorOpUpload::accept(runtime::TensorNodeExecutor & node_executor,
                           runtime::TensorOpExecHandle * exec_handle)
{
 return node_executor.execute(*this,exec_handle);
}

std::unique_ptr<TensorOperation> TensorOpUpload::createNew()
{
 return std::unique_ptr<TensorOperation>(new TensorOpUpload());
}

bool TensorOpUpload::resetMPICommunicator(const MPICommProxy & intra_comm)
{
 intra_comm_ = intra_comm;
 return true;
}

const MPICommProxy & TensorOpUpload::getMPICommunicator() const
{
 return intra_comm_;
}

bool TensorOpUpload::resetRemoteProcessRank(unsigned int rank)
{
 remote_rank_ = static_cast<int>(rank);
 return true;
}

int TensorOpUpload::getRemoteProcessRank() const
{
 return remote_rank_;
}

bool TensorOpUpload::resetMessageTag(int tag)
{
 message_tag_ = tag;
 return true;
}

int TensorOpUpload::getMessageTag() const
{
 return message_tag_;
}

} //namespace numerics

} //namespace exatn
