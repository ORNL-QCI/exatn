/** ExaTN::Numerics: Tensor network builder: MPS: Matrix Product State
REVISION: 2019/07/12

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "network_builder_mps.hpp"
#include "tensor_network.hpp"

namespace exatn{

namespace numerics{

NetworkBuilderMPS::NetworkBuilderMPS():
 max_bond_dim_(1)
{
}


bool NetworkBuilderMPS::getParameter(const std::string & name, long long * value) const
{
 bool found = true;
 if(name == "max_bond_dim"){
  *value = max_bond_dim_;
 }else{
  found = false;
 }
 return found;
}


bool NetworkBuilderMPS::setParameter(const std::string & name, long long value)
{
 bool found = true;
 if(name == "max_bond_dim"){
  max_bond_dim_ = value;
 }else{
  found = false;
 }
 return found;
}


void NetworkBuilderMPS::build(TensorNetwork & network)
{
 //`Finish
 return;
}


std::unique_ptr<NetworkBuilder> NetworkBuilderMPS::createNew()
{
 return std::unique_ptr<NetworkBuilder>(new NetworkBuilderMPS());
}

} //namespace numerics

} //namespace exatn
