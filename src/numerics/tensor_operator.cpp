/** ExaTN::Numerics: Tensor operator
REVISION: 2022/08/30

Copyright (C) 2018-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_operator.hpp"
#include "tensor_range.hpp"

namespace exatn{

namespace numerics{

TensorOperator::TensorOperator(const std::string & name,
                               std::shared_ptr<TensorNetwork> network,
                               const std::vector<std::pair<unsigned int, unsigned int>> & ket_pairing,
                               const std::vector<std::pair<unsigned int, unsigned int>> & bra_pairing,
                               const std::complex<double> coefficient):
 name_(name)
{
 auto success = appendComponent(network,ket_pairing,bra_pairing,coefficient);
 assert(success);
}


TensorOperator::TensorOperator(const std::string & name,
                               std::shared_ptr<TensorNetwork> ket_network,
                               std::shared_ptr<TensorNetwork> bra_network,
                               const std::vector<std::pair<unsigned int, unsigned int>> & ket_pairing,
                               const std::vector<std::pair<unsigned int, unsigned int>> & bra_pairing,
                               const std::complex<double> coefficient):
 name_(name)
{
 auto success = appendComponent(ket_network,bra_network,ket_pairing,bra_pairing,coefficient);
 assert(success);
}


bool TensorOperator::appendComponent(std::shared_ptr<TensorNetwork> network, //in: tensor network (or single tensor as a tensor network)
     const std::vector<std::pair<unsigned int, unsigned int>> & ket_pairing, //in: ket pairing: Global tensor mode id <-- Output tensor leg
     const std::vector<std::pair<unsigned int, unsigned int>> & bra_pairing, //in: bra pairing: Global tensor mode id <-- Output tensor leg
     const std::complex<double> coefficient)                                 //in: expansion coefficient
{
 assert(network);
 auto output_tensor = network->getTensor(0);
 const auto output_tensor_rank = output_tensor->getRank();
 assert(ket_pairing.size() + bra_pairing.size() == output_tensor_rank);
 components_.emplace_back(OperatorComponent{network,ket_pairing,bra_pairing,coefficient});
 return true;
}


bool TensorOperator::appendComponent(std::shared_ptr<Tensor> tensor,         //in: tensor
     const std::vector<std::pair<unsigned int, unsigned int>> & ket_pairing, //in: ket pairing: Global tensor mode id <-- Tensor leg
     const std::vector<std::pair<unsigned int, unsigned int>> & bra_pairing, //in: bra pairing: Global tensor mode id <-- Tensor leg
     const std::complex<double> coefficient)                                 //in: expansion coefficient
{
 assert(tensor);
 bool appended = false;
 const auto tensor_rank = tensor->getRank();
 auto output_tensor = std::make_shared<Tensor>(*tensor);
 output_tensor->rename("_"+(tensor->getName())+"_");
 std::vector<TensorLeg> legs(tensor_rank,TensorLeg{0,0});
 for(unsigned int i = 0; i < tensor_rank; ++i) legs[i] = TensorLeg{1,i};
 auto network = makeSharedTensorNetwork(tensor->getName(),output_tensor,legs);
 for(unsigned int i = 0; i < tensor_rank; ++i) legs[i] = TensorLeg{0,i};
 appended = network->placeTensor(1,tensor,legs);
 if(appended){
  appended = network->finalize();
  if(appended){
   appended = appendComponent(network,ket_pairing,bra_pairing,coefficient);
  }else{
   std::cout << "#ERROR(exatn::numerics::TensorOperator::appendComponent): Unable to finalize the intermediate tensor network!"
             << std::endl;
  }
 }else{
  std::cout << "#ERROR(exatn::numerics::TensorOperator::appendComponent): Unable to build a tensor network from the given tensor!"
            << std::endl;
 }
 return appended;
}


bool TensorOperator::appendSymmetrizeComponent(std::shared_ptr<TensorNetwork> network,
     const std::vector<unsigned int> & ket_pairing,
     const std::vector<unsigned int> & bra_pairing,
     unsigned int ket_space_rank,
     unsigned int bra_space_rank,
     const std::complex<double> coefficient,
     bool antisymmetrize)
{
 bool success = true;
 auto ket_rank = ket_pairing.size();
 auto bra_rank = bra_pairing.size();
 assert(ket_rank + bra_rank == network->getRank());
 assert(ket_rank <= ket_space_rank);
 assert(bra_rank <= bra_space_rank);
 std::vector<std::pair<unsigned int, unsigned int>> ket_pairs(ket_rank);
 std::vector<std::pair<unsigned int, unsigned int>> bra_pairs(bra_rank);
 std::vector<DimExtent> ket_range(ket_rank,static_cast<DimExtent>(ket_space_rank));
 std::vector<DimExtent> bra_range(bra_rank,static_cast<DimExtent>(bra_space_rank));
 TensorRange bra_legs(bra_range);
 bool bra_not_over = true;
 while(bra_not_over){
  if(bra_legs.increasingOrder()){
   for(int i = 0; i < bra_rank; ++i) bra_pairs[i] = {static_cast<unsigned int>(bra_legs[i]), bra_pairing[i]};
   double phase_bra = 1.0;
   if(antisymmetrize){
    for(int i = 0; i < bra_rank; ++i){
     if(((bra_legs[i] - i) % 2) != 0) phase_bra = -phase_bra;
    }
   }
   TensorRange ket_legs(ket_range);
   bool ket_not_over = true;
   while(ket_not_over){
    if(ket_legs.increasingOrder()){
     for(int i = 0; i < ket_rank; ++i) ket_pairs[i] = {static_cast<unsigned int>(ket_legs[i]), ket_pairing[i]};
     double phase_ket = 1.0;
     if(antisymmetrize){
      for(int i = 0; i < ket_rank; ++i){
       if(((ket_legs[i] - i) % 2) != 0) phase_ket = -phase_ket;
      }
     }
     success = appendComponent(network,ket_pairs,bra_pairs,coefficient*std::complex<double>{phase_bra*phase_ket,0.0});
     if(!success) return success;
    }
    ket_not_over = ket_legs.next();
   }
  }
  bra_not_over = bra_legs.next();
 }
 return success;
}


bool TensorOperator::appendSymmetrizeComponent(std::shared_ptr<Tensor> tensor,
     const std::vector<unsigned int> & ket_pairing,
     const std::vector<unsigned int> & bra_pairing,
     unsigned int ket_space_rank,
     unsigned int bra_space_rank,
     const std::complex<double> coefficient,
     bool antisymmetrize)
{
 bool success = true;
 auto ket_rank = ket_pairing.size();
 auto bra_rank = bra_pairing.size();
 assert(ket_rank + bra_rank == tensor->getRank());
 assert(ket_rank <= ket_space_rank);
 assert(bra_rank <= bra_space_rank);
 std::vector<std::pair<unsigned int, unsigned int>> ket_pairs(ket_rank);
 std::vector<std::pair<unsigned int, unsigned int>> bra_pairs(bra_rank);
 std::vector<DimExtent> ket_range(ket_rank,static_cast<DimExtent>(ket_space_rank));
 std::vector<DimExtent> bra_range(bra_rank,static_cast<DimExtent>(bra_space_rank));
 TensorRange bra_legs(bra_range);
 bool bra_not_over = true;
 while(bra_not_over){
  if(bra_legs.increasingOrder()){
   for(int i = 0; i < bra_rank; ++i) bra_pairs[i] = {static_cast<unsigned int>(bra_legs[i]), bra_pairing[i]};
   double phase_bra = 1.0;
   if(antisymmetrize){
    for(int i = 0; i < bra_rank; ++i){
     if(((bra_legs[i] - i) % 2) != 0) phase_bra = -phase_bra;
    }
   }
   TensorRange ket_legs(ket_range);
   bool ket_not_over = true;
   while(ket_not_over){
    if(ket_legs.increasingOrder()){
     for(int i = 0; i < ket_rank; ++i) ket_pairs[i] = {static_cast<unsigned int>(ket_legs[i]), ket_pairing[i]};
     double phase_ket = 1.0;
     if(antisymmetrize){
      for(int i = 0; i < ket_rank; ++i){
       if(((ket_legs[i] - i) % 2) != 0) phase_ket = -phase_ket;
      }
     }
     success = appendComponent(tensor,ket_pairs,bra_pairs,coefficient*std::complex<double>{phase_bra*phase_ket,0.0});
     if(!success) return success;
    }
    ket_not_over = ket_legs.next();
   }
  }
  bra_not_over = bra_legs.next();
 }
 return success;
}


bool TensorOperator::appendComponent(std::shared_ptr<TensorNetwork> ket_network,
                                     std::shared_ptr<TensorNetwork> bra_network,
                                     const std::vector<std::pair<unsigned int, unsigned int>> & ket_pairing,
                                     const std::vector<std::pair<unsigned int, unsigned int>> & bra_pairing,
                                     const std::complex<double> coefficient)
{
 auto shifted_bra_pairing = bra_pairing;
 const auto shift = ket_network->getRank();
 for(auto & pairing: shifted_bra_pairing) pairing.second += shift;
 auto combined_network = makeSharedTensorNetwork(*ket_network,true,ket_network->getName());
 combined_network->conjugate();
 auto success = combined_network->appendTensorNetwork(TensorNetwork(*bra_network,true,bra_network->getName()),{});
 assert(success);
 return appendComponent(combined_network,ket_pairing,shifted_bra_pairing,coefficient);
}


bool TensorOperator::deleteComponent(std::size_t component_num)
{
 if(component_num >= components_.size()) return false;
 components_.erase(components_.cbegin()+component_num);
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


void TensorOperator::rescale(std::complex<double> scaling_factor)
{
 for(auto & component: components_) component.coefficient *= scaling_factor;
 return;
}


std::vector<std::complex<double>> TensorOperator::getCoefficients() const
{
 std::vector<std::complex<double>> coefs(components_.size(),{0.0,0.0});
 std::size_t i = 0;
 for(const auto & component: components_) coefs[i++] = component.coefficient;
 return coefs;
}


void TensorOperator::printIt() const
{
 std::cout << "TensorNetworkOperator(" << this->getName()
           << ")[size = " << this->getNumComponents() << "]{" << std::endl;
 std::size_t i = 0;
 for(const auto & component: components_){
  std::cout << "Component " << i++ << ": "
            << std::scientific << component.coefficient << std::endl;
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


std::shared_ptr<numerics::TensorOperator> combineTensorOperators(const numerics::TensorOperator & operator1,
                                                                 const numerics::TensorOperator & operator2)
{
 auto operator_result = makeSharedTensorOperator(operator1.getName() + "+" + operator2.getName());
 for(auto component = operator1.cbegin(); component != operator1.cend(); ++component){
  auto success = operator_result->appendComponent(component->network,
                                   component->ket_legs,component->bra_legs,component->coefficient);
  assert(success);
 }
 for(auto component = operator2.cbegin(); component != operator2.cend(); ++component){
  auto success = operator_result->appendComponent(component->network,
                                   component->ket_legs,component->bra_legs,component->coefficient);
  assert(success);
 }
 return operator_result;
}

} //namespace exatn
