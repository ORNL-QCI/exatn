/** ExaTN::Numerics: Tensor operation: Destroys a tensor
REVISION: 2021/07/15

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "exatn_service.hpp"

#include "tensor_op_destroy.hpp"

#include "tensor_node_executor.hpp"

namespace exatn{

namespace numerics{

TensorOpDestroy::TensorOpDestroy():
 TensorOperation(TensorOpCode::DESTROY,1,0,1,{0})
{
}

bool TensorOpDestroy::isSet() const
{
 return (this->getNumOperandsSet() == this->getNumOperands());
}

int TensorOpDestroy::accept(runtime::TensorNodeExecutor & node_executor,
                            runtime::TensorOpExecHandle * exec_handle)
{
 return node_executor.execute(*this,exec_handle);
}

std::unique_ptr<TensorOperation> TensorOpDestroy::createNew()
{
 return std::unique_ptr<TensorOperation>(new TensorOpDestroy());
}

std::size_t TensorOpDestroy::decompose(const TensorMapper & tensor_mapper)
{
 simple_operations_.clear();
 auto tensor0 = getTensorOperand(0);
 if(tensor0->isComposite()){
  auto composite_tensor0 = castTensorComposite(tensor0);
  const auto num_subtensors = composite_tensor0->getNumSubtensors();
  for(auto subtensor_iter = composite_tensor0->begin(); subtensor_iter != composite_tensor0->end(); ++subtensor_iter){
   if(tensor_mapper.isLocalSubtensor(subtensor_iter->first,num_subtensors)){
    simple_operations_.emplace_back(std::move(TensorOpDestroy::createNew()));
    auto & op = simple_operations_.back();
    op->setTensorOperand(subtensor_iter->second);
   }
  }
 }
 return simple_operations_.size();
}

} //namespace numerics

} //namespace exatn
