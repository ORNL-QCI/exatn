/** ExaTN::Numerics: Tensor operation: Uploads remote tensor data
REVISION: 2021/07/28

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

void TensorOpUpload::printIt() const
{
 std::cout << "TensorOperation(opcode=" << static_cast<int>(opcode_) << ")[id=" << id_ << "]{" << std::endl;
 if(pattern_.length() > 0) std::cout << " " << pattern_ << std::endl;
 for(const auto & operand: operands_){
  const auto & tensor = std::get<0>(operand);
  if(tensor != nullptr){
   std::cout << " ";
   tensor->printIt();
   std::cout << std::endl;
  }else{
   std::cout << "#ERROR(exatn::TensorOperation::printIt): Tensor operand is NULL!" << std::endl << std::flush;
   assert(false);
  }
 }
 for(const auto & scalar: scalars_){
  std::cout << " " << scalar;
 }
 if(scalars_.size() > 0) std::cout << std::endl;
 std::cout << " Remote process rank = " << remote_rank_ << ": Message tag = " << message_tag_ << " (upload)" << std::endl;
 std::cout << " GWord estimate = " << std::scientific << this->getWordEstimate()/1e9 << std::endl;
 std::cout << "}" << std::endl << std::flush;
 return;
}

void TensorOpUpload::printItFile(std::ofstream & output_file) const
{
 output_file << "TensorOperation(opcode=" << static_cast<int>(opcode_) << ")[id=" << id_ << "]{" << std::endl;
 if(pattern_.length() > 0) output_file << " " << pattern_ << std::endl;
 for(const auto & operand: operands_){
  const auto & tensor = std::get<0>(operand);
  if(tensor != nullptr){
   output_file << " ";
   tensor->printItFile(output_file);
   output_file << std::endl;
  }else{
   std::cout << "#ERROR(exatn::TensorOperation::printItFile): Tensor operand is NULL!" << std::endl << std::flush;
   assert(false);
  }
 }
 for(const auto & scalar: scalars_){
  output_file << " " << scalar;
 }
 if(scalars_.size() > 0) output_file << std::endl;
 output_file << " Remote process rank = " << remote_rank_ << ": Message tag = " << message_tag_ << " (upload)" << std::endl;
 output_file << " GWord estimate = " << std::scientific << this->getWordEstimate()/1e9 << std::endl;
 output_file << "}" << std::endl;
 //output_file.flush();
 return;
}

std::size_t TensorOpUpload::decompose(const TensorMapper & tensor_mapper)
{
 assert(false);
 //`Implement
 return 0;
}

} //namespace numerics

} //namespace exatn
