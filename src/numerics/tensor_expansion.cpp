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
  assert(output_tensor_rank == first_tensor_rank);
  auto congruent = output_tensor->isCongruentTo(*first_tensor);
  assert(congruent);
  const auto * output_legs = network->getTensorConnections(0);
  const auto * first_legs = components_[0].network->getTensorConnections(0);
  congruent = tensorLegsAreCongruent(output_legs,first_legs);
  assert(congruent);
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

} //namespace numerics

} //namespace exatn
