/** ExaTN::Numerics: Tensor operation: Contracts two tensors and accumulates the result into another tensor
REVISION: 2021/08/18

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "exatn_service.hpp"

#include "tensor_op_contract.hpp"

#include "tensor_node_executor.hpp"

#include <cmath>

namespace exatn{

namespace numerics{

TensorOpContract::TensorOpContract():
 TensorOperation(TensorOpCode::CONTRACT,3,2,1+0*2+0*4,{0,1,2}),
 accumulative_(true)
{
 this->setScalar(0,std::complex<double>{1.0,0.0}); //default alpha prefactor
 this->setScalar(1,std::complex<double>{1.0,0.0}); //default beta prefactor (accumulative)
}

bool TensorOpContract::isSet() const
{
 return (this->getNumOperandsSet() == this->getNumOperands() && this->getIndexPattern().length() > 0);
}

int TensorOpContract::accept(runtime::TensorNodeExecutor & node_executor,
                             runtime::TensorOpExecHandle * exec_handle)
{
 return node_executor.execute(*this,exec_handle);
}

double TensorOpContract::getFlopEstimate() const
{
 if(this->isSet()){
  double vol0 = static_cast<double>(this->getTensorOperand(0)->getVolume());
  double vol1 = static_cast<double>(this->getTensorOperand(1)->getVolume());
  double vol2 = static_cast<double>(this->getTensorOperand(2)->getVolume());
  return std::sqrt(vol0*vol1*vol2); //FMA flops (without FMA factor)
 }
 return 0.0;
}

std::unique_ptr<TensorOperation> TensorOpContract::createNew()
{
 return std::unique_ptr<TensorOperation>(new TensorOpContract());
}

void TensorOpContract::resetAccumulative(bool accum)
{
 accumulative_ = accum;
 return;
}

void TensorOpContract::introduceOptTemporaries(unsigned int num_processes,
                                               std::size_t mem_per_process,
                                               const std::vector<PosIndexLabel> & left_indices,
                                               const std::vector<PosIndexLabel> & right_indices,
                                               const std::vector<PosIndexLabel> & contr_indices,
                                               const std::vector<PosIndexLabel> & hyper_indices)
{
 bool success = true;
 index_info_ = std::make_shared<IndexInfo>(this);
 for(const auto & lbl: left_indices){
  success = index_info_->appendIndex(lbl); assert(success);
 }
 for(const auto & lbl: right_indices){
  success = index_info_->appendIndex(lbl); assert(success);
 }
 for(const auto & lbl: contr_indices){
  success = index_info_->appendIndex(lbl); assert(success);
 }
 for(const auto & lbl: hyper_indices){
  success = index_info_->appendIndex(lbl); assert(success);
 }
 //index_info_->printIt(); //debug
 
 std::abort(); //debug
 return;
}

void TensorOpContract::introduceOptTemporaries(unsigned int num_processes, std::size_t mem_per_process)
{
 std::vector<std::string> tensors;
 std::vector<PosIndexLabel> left_inds, right_inds, contr_inds, hyper_inds;
 auto parsed = parse_tensor_contraction(getIndexPattern(),tensors,left_inds,right_inds,contr_inds,hyper_inds);
 if(!parsed){
  std::cout << "#ERROR(TensorOpContract:introduceOptTemporaries): Invalid tensor contraction specification: "
            << getIndexPattern() << std::endl << std::flush;
  assert(false);
 }
 return introduceOptTemporaries(num_processes,mem_per_process,left_inds,right_inds,contr_inds,hyper_inds);
}

std::size_t TensorOpContract::decompose(const TensorMapper & tensor_mapper)
{
 if(this->isComposite()){
  if(simple_operations_.empty()){
   //Identify parallel configuration:
   const auto num_procs = tensor_mapper.getNumProcesses();
   const auto proc_rank = tensor_mapper.getProcessRank();
   const auto & intra_comm = tensor_mapper.getMPICommProxy();
   //Proceed with decomposition:
   assert(index_info_);
   
  }
 }
 return simple_operations_.size();
}

} //namespace numerics

} //namespace exatn
