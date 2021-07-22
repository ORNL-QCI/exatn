/** ExaTN::Numerics: Tensor operation: Adds a tensor to another tensor
REVISION: 2021/07/22

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "exatn_service.hpp"

#include "tensor_op_add.hpp"

#include "tensor_node_executor.hpp"

namespace exatn{

namespace numerics{

TensorOpAdd::TensorOpAdd():
 TensorOperation(TensorOpCode::ADD,2,1,1+0*2,{0,1})
{
 this->setScalar(0,std::complex<double>{1.0,0.0}); //default alpha prefactor
}

bool TensorOpAdd::isSet() const
{
 return (this->getNumOperandsSet() == this->getNumOperands() && this->getIndexPattern().length() > 0);
}

int TensorOpAdd::accept(runtime::TensorNodeExecutor & node_executor,
                        runtime::TensorOpExecHandle * exec_handle)
{
 return node_executor.execute(*this,exec_handle);
}

double TensorOpAdd::getFlopEstimate() const
{
 if(this->isSet()) return static_cast<double>(this->getTensorOperand(0)->getVolume()); //FMA flops (without FMA factor)
 return 0.0;
}

std::unique_ptr<TensorOperation> TensorOpAdd::createNew()
{
 return std::unique_ptr<TensorOperation>(new TensorOpAdd());
}

std::size_t TensorOpAdd::decompose(const TensorMapper & tensor_mapper)
{
 if(this->isComposite()){
  if(simple_operations_.empty()){
   //Prepare composite tensor operands:
   auto tensor0 = getTensorOperand(0);
   auto tensor1 = getTensorOperand(1);
   auto comp_tens0 = castTensorComposite(tensor0);
   auto comp_tens1 = castTensorComposite(tensor1);
   assert(comp_tens0 || comp_tens1);
   const auto num_procs = tensor_mapper.getNumProcesses();
   const auto proc_rank = tensor_mapper.getProcessRank();
   const auto tens_elem_type0 = tensor0->getElementType();
   const auto tens_elem_type1 = tensor1->getElementType();
   std::map<unsigned long long, std::shared_ptr<Tensor>> self_composite0 {{0,tensor0}};
   std::map<unsigned long long, std::shared_ptr<Tensor>> self_composite1 {{0,tensor1}};
   auto beg0 = self_composite0.begin();
   auto end0 = self_composite0.end();
   unsigned long long num_subtensors0 = 1;
   if(comp_tens0){
    beg0 = comp_tens0->begin();
    end0 = comp_tens0->end();
    num_subtensors0 = comp_tens0->getNumSubtensors();
   }
   auto beg1 = self_composite1.begin();
   auto end1 = self_composite1.end();
   unsigned long long num_subtensors1 = 1;
   if(comp_tens1){
    beg1 = comp_tens1->begin();
    end1 = comp_tens1->end();
    num_subtensors1 = comp_tens1->getNumSubtensors();
   }
   //Send local tensor slices to remote processes:
   for(auto subtens1 = beg1; subtens1 != end1; ++subtens1){
    if(tensor_mapper.isLocalSubtensor(*(subtens1->second))){
     for(auto subtens0 = beg0; subtens0 != end0; ++subtens0){
      auto slice = makeSharedTensorIntersection("_",*(subtens0->second),*(subtens1->second));
      if(slice){
       slice->rename();
       simple_operations_.emplace_back(std::move(TensorOpCreate::createNew()));
       auto & op = simple_operations_.back();
       op->setTensorOperand(slice);
       std::dynamic_pointer_cast<TensorOpCreate>(op)->resetTensorElementType(tens_elem_type1);

      }
     }
    }
   }
   //Receive necessary tensor slices from remote processes:
   
  }
 }
 return simple_operations_.size();
}

} //namespace numerics

} //namespace exatn
