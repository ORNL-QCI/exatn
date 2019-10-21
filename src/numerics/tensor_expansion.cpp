/** ExaTN::Numerics: Tensor network expansion
REVISION: 2019/10/21

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_expansion.hpp"

namespace exatn{

namespace numerics{

bool TensorExpansion::appendComponent(std::shared_ptr<TensorNetwork> network, //in: tensor network
                                      const std::complex<double> coefficient) //in: expansion coefficient
{
 auto output_tensor = network->getTensor(0);
 const auto output_tensor_rank = output_tensor->getRank();
 if(!(components_.empty())){
  auto first_tensor = components_[0].network->getTensor(0);
  const auto first_tensor_rank = first_tensor->getRank();
  if(first_tensor_rank != output_tensor_rank){
   std::cout << "#ERROR(exatn::numerics::TensorExpansion::appendComponent): Tensor rank mismatch: "
             << first_tensor_rank << " versus " << output_tensor_rank << std::endl;
   assert(false);
  }
  auto congruent = output_tensor->isCongruentTo(*first_tensor);
  if(!congruent){
   std::cout << "#ERROR(exatn::numerics::TensorExpansion::appendComponent): Tensor shape mismatch!" << std::endl;
   assert(false);
  }
  const auto * output_legs = network->getTensorConnections(0);
  const auto * first_legs = components_[0].network->getTensorConnections(0);
  congruent = tensorLegsAreCongruent(output_legs,first_legs);
  if(!congruent){
   std::cout << "#ERROR(exatn::numerics::TensorExpansion::appendComponent): Tensor leg direction mismatch!" << std::endl;
   assert(false);
  }
 }
 components_.emplace_back(ExpansionComponent{network,coefficient});
 return true;
}


void TensorExpansion::conjugate()
{
 for(auto & component: components_){
  component.network->conjugate();
  component.coefficient = std::conj(component.coefficient);
 }
 ket_ = !ket_;
 return;
}


bool TensorExpansion::applyOperator(const TensorOperator & tensor_operator) //in: tensor network operator
{
 //`Finish
 return true;
}


bool TensorExpansion::formInnerProduct(const TensorExpansion & dual_expansion) //in: tensor network expansion from the dual tensor space
{
 if((this->isKet() && dual_expansion.isKet()) || (this->isBra() && dual_expansion.isBra())){
  std::cout << "#ERROR(exatn::numerics::TensorExpansion::formInnerProduct): Invalid duality!" << std::endl;
  return false;
 }
 //`Finish
 return true;
}


bool TensorExpansion::formInnerProduct(const TensorOperator & tensor_operator, //in: tensor network operator
                                       const TensorExpansion & dual_expansion) //in: tensor network expansion from the dual tensor space
{
 auto success = this->applyOperator(tensor_operator);
 if(success) success = this->formInnerProduct(dual_expansion);
 return success;
}

} //namespace numerics

} //namespace exatn
