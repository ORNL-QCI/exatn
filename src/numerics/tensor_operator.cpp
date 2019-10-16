/** ExaTN::Numerics: Tensor operator
REVISION: 2019/10/16

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_operator.hpp"

namespace exatn{

namespace numerics{

bool TensorOperator::appendComponent(std::shared_ptr<TensorNetwork> network, //in: tensor network (or single tensor as a tensor network)
     const std::vector<std::pair<unsigned int, unsigned int>> & ket_pairing, //in: ket pairing: Output tensor leg --> global tensor mode id
     const std::vector<std::pair<unsigned int, unsigned int>> & bra_pairing, //in: bra pairing: Output tensor leg --> global tensor mode id
     const std::complex<double> coefficient)                                 //in: expansion coefficient
{
 auto output_tensor = network->getTensor(0);
 const auto output_tensor_rank = output_tensor->getRank();
 assert(ket_pairing.size() + bra_pairing.size() == output_tensor_rank);
 components_.emplace_back(OperatorComponent{network,ket_pairing,bra_pairing,coefficient});
 return true;
}

void TensorOperator::conjugate()
{
 for(auto & component: components_){
  component.network->conjugate();
  component.ket_legs.swap(component.bra_legs);
  component.coefficient = std::conj(component.coefficient);
 }
 return;
}

} //namespace numerics

} //namespace exatn
