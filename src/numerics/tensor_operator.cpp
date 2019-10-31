/** ExaTN::Numerics: Tensor operator
REVISION: 2019/10/31

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_operator.hpp"

namespace exatn{

namespace numerics{

bool TensorOperator::appendComponent(std::shared_ptr<TensorNetwork> network, //in: tensor network (or single tensor as a tensor network)
     const std::vector<std::pair<unsigned int, unsigned int>> & ket_pairing, //in: ket pairing: Global tensor mode id <-- Output tensor leg
     const std::vector<std::pair<unsigned int, unsigned int>> & bra_pairing, //in: bra pairing: Global tensor mode id <-- Output tensor leg
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


void TensorOperator::printIt() const
{
 std::cout << "TensorNetworkOperator(" << this->getName()
           << ")[rank = " << this->getRank()
           << ", size = " << this->getNumComponents() << "]{" << std::endl;
 std::size_t i = 0;
 for(const auto & component: components_){
  std::cout << "Component " << i++ << ": " << component.coefficient << std::endl;
  std::cout << "Ket legs { ";
  for(const auto & leg: component.ket_legs) std::cout << "{" << leg.second << "->" << leg.first << "}";
  std::cout << " }" << std::endl;
  std::cout << "Bra legs { ";
  for(const auto & leg: component.bra_legs) std::cout << "{" << leg.second << "->" << leg.first << "}";
  std::cout << " }" << std::endl;
  component.network->printIt();
 }
 std::cout << "}" << std::endl;
 return;
}

} //namespace numerics

} //namespace exatn
