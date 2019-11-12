/** ExaTN::Numerics: Tensor network expansion
REVISION: 2019/10/31

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_expansion.hpp"

#include <algorithm>

#include <cassert>

namespace exatn{

namespace numerics{

TensorExpansion::TensorExpansion(const TensorExpansion & expansion,       //in: tensor network expansion in some tensor space
                                 const TensorOperator & tensor_operator): //in: tensor network operator
 ket_(expansion.isKet())
{
 bool appended;
 for(auto term = expansion.cbegin(); term != expansion.cend(); ++term){
  for(auto oper = tensor_operator.cbegin(); oper != tensor_operator.cend(); ++oper){
   auto product = std::make_shared<TensorNetwork>(*(term->network));
   if(ket_){
    appended = product->appendTensorNetwork(TensorNetwork(*(oper->network)),oper->ket_legs);
    assert(appended);
    appended = reorderProductLegs(*product,oper->bra_legs);
    assert(appended);
   }else{
    appended = product->appendTensorNetwork(TensorNetwork(*(oper->network)),oper->bra_legs);
    assert(appended);
    appended = reorderProductLegs(*product,oper->ket_legs);
    assert(appended);
   }
   product->rename(oper->network->getName() + "*" + term->network->getName());
   appended = this->appendComponent(product,(oper->coefficient)*(term->coefficient));
   assert(appended);
  }
 }
}


TensorExpansion::TensorExpansion(const TensorExpansion & left_expansion,  //in: tensor network expansion in some tensor space
                                 const TensorExpansion & right_expansion) //in: tensor network expansion from the same or dual space
{
 if((left_expansion.isKet() && right_expansion.isKet()) || (left_expansion.isBra() && right_expansion.isBra())){
  constructDirectProductTensorExpansion(left_expansion,right_expansion);
  ket_ = left_expansion.isKet();
 }else{
  constructInnerProductTensorExpansion(left_expansion,right_expansion);
  ket_ = true; //inner product tensor expansion is formally marked as ket but it is irrelevant
 }
}


TensorExpansion::TensorExpansion(const TensorExpansion & left_expansion,  //in: tensor network expansion in some tensor space
                                 const TensorExpansion & right_expansion, //in: tensor network expansion from the dual tensor space
                                 const TensorOperator & tensor_operator)  //in: tensor network operator
{
 constructInnerProductTensorExpansion(left_expansion,TensorExpansion(right_expansion,tensor_operator));
 ket_ = true; //inner product tensor expansion is formally marked as ket but it is irrelevant
}


bool TensorExpansion::appendComponent(std::shared_ptr<TensorNetwork> network, //in: tensor network
                                      const std::complex<double> coefficient) //in: expansion coefficient
{
 auto output_tensor = network->getTensor(0);
 const auto output_tensor_rank = output_tensor->getRank();
 //Check validity:
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
 //Append new component:
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


void TensorExpansion::printIt() const
{
 if(ket_){
  std::cout << "TensorNetworkExpansion()[ket rank = " << this->getRank()
            << ", size = " << this->getNumComponents() << "]{" << std::endl;
 }else{
  std::cout << "TensorNetworkExpansion()[bra rank = " << this->getRank()
            << ", size = " << this->getNumComponents() << "]{" << std::endl;
 }
 std::size_t i = 0;
 for(const auto & component: components_){
  std::cout << "Component " << i++ << ": " << component.coefficient << std::endl;
  component.network->printIt();
 }
 std::cout << "}" << std::endl;
 return;
}


void TensorExpansion::constructDirectProductTensorExpansion(const TensorExpansion & left_expansion,
                                                            const TensorExpansion & right_expansion)
{
 if(left_expansion.getNumComponents() == 0 || right_expansion.getNumComponents() == 0){
  std::cout << "#ERROR(exatn::numerics::TensorExpansion::constructDirectProductTensorExpansion): Empty input expansion!"
            << std::endl;
  assert(false);
 }
 bool appended;
 std::vector<std::pair<unsigned int, unsigned int>> pairing;
 for(auto left = left_expansion.cbegin(); left != left_expansion.cend(); ++left){
  for(auto right = right_expansion.cbegin(); right != right_expansion.cend(); ++right){
   auto product = std::make_shared<TensorNetwork>(*(left->network));
   appended = product->appendTensorNetwork(TensorNetwork(*(right->network)),pairing);
   assert(appended);
   product->rename(left->network->getName() + "*" + right->network->getName());
   appended = this->appendComponent(product,(left->coefficient)*(right->coefficient));
   assert(appended);
  }
 }
 return;
}


void TensorExpansion::constructInnerProductTensorExpansion(const TensorExpansion & left_expansion,
                                                           const TensorExpansion & right_expansion)
{
 if(left_expansion.getNumComponents() == 0 || right_expansion.getNumComponents() == 0){
  std::cout << "#ERROR(exatn::numerics::TensorExpansion::constructInnerProductTensorExpansion): Empty input expansion!"
            << std::endl;
  assert(false);
 }
 auto rank = left_expansion.cbegin()->network->getRank();
 assert(rank > 0);
 bool appended;
 std::vector<std::pair<unsigned int, unsigned int>> pairing(rank);
 for(unsigned int i = 0; i < rank; ++i) pairing[i] = {i,i};
 for(auto left = left_expansion.cbegin(); left != left_expansion.cend(); ++left){
  for(auto right = right_expansion.cbegin(); right != right_expansion.cend(); ++right){
   assert(right->network->getRank() == rank);
   auto product = std::make_shared<TensorNetwork>(*(right->network));
   appended = product->appendTensorNetwork(TensorNetwork(*(left->network)),pairing);
   assert(appended);
   product->rename(left->network->getName() + "*" + right->network->getName());
   appended = this->appendComponent(product,(left->coefficient)*(right->coefficient));
   assert(appended);
  }
 }
 return;
}


bool TensorExpansion::reorderProductLegs(TensorNetwork & network,
     const std::vector<std::pair<unsigned int, unsigned int>> & new_legs)
{
 auto network_rank = network.getRank();
 auto num_new_legs = new_legs.size();
 assert(num_new_legs <= network_rank);
 if(num_new_legs > 0){
  auto sorted_new_legs = new_legs;
  std::sort(sorted_new_legs.begin(),sorted_new_legs.end(),
            [](const std::pair<unsigned int, unsigned int> & item0,
               const std::pair<unsigned int, unsigned int> & item1){
             return item0.second < item1.second; //order legs by their ids in the tensor operator:
            }                                    //this is how they were appended into the network
           );
  unsigned int n = (network_rank - num_new_legs); //first new leg id in the network
  for(auto & leg: sorted_new_legs) leg.second = n++; //second is now the position of the leg in the network
  std::sort(sorted_new_legs.begin(),sorted_new_legs.end(),
            [](const std::pair<unsigned int, unsigned int> & item0,
               const std::pair<unsigned int, unsigned int> & item1){
             return item0.first < item1.first; //order legs by their global ids
            }
           );
  std::vector<unsigned int> new_order(network_rank);
  unsigned int i = 0, j = 0;
  auto new_leg = sorted_new_legs.begin();
  while(i < network_rank){
   if(new_leg != sorted_new_legs.end()){
    if(new_leg->first == i){
     new_order[i++] = new_leg->second;
     ++new_leg;
    }else{
     new_order[i++] = j++;
    }
   }else{
    new_order[i++] = j++;
   }
  }
  assert(j == network_rank - num_new_legs);
  return network.reorderOutputModes(new_order);
 }
 return true;
}

} //namespace numerics

} //namespace exatn
