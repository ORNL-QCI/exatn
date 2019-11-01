/** ExaTN::Numerics: Tensor network builder: Tree: Tree Tensor Network
REVISION: 2019/11/01

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "network_builder_tree.hpp"
#include "tensor_network.hpp"

namespace exatn{

namespace numerics{

NetworkBuilderTree::NetworkBuilderTree():
 max_bond_dim_(1), arity_(2)
{
}


bool NetworkBuilderTree::getParameter(const std::string & name, long long * value) const
{
 bool found = true;
 if(name == "max_bond_dim"){
  *value = max_bond_dim_;
 }else if(name == "arity"){
  *value = arity_;
 }else{
  found = false;
 }
 return found;
}


bool NetworkBuilderTree::setParameter(const std::string & name, long long value)
{
 bool found = true;
 if(name == "max_bond_dim"){
  max_bond_dim_ = value;
 }else if(name == "arity"){
  arity_ = value;
 }else{
  found = false;
 }
 return found;
}


void NetworkBuilderTree::build(TensorNetwork & network)
{
 //`Finish
 return;
}


std::unique_ptr<NetworkBuilder> NetworkBuilderTree::createNew()
{
 return std::unique_ptr<NetworkBuilder>(new NetworkBuilderTree());
}

} //namespace numerics

} //namespace exatn
