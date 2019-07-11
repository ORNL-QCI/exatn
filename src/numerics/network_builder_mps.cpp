/** ExaTN::Numerics: Tensor network builder: MPS: Matrix Product State
REVISION: 2019/07/11

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "network_builder_mps.hpp"

namespace exatn{

namespace numerics{

NetworkBuilderMPS::NetworkBuilderMPS():
 num_sites_(1)
{
}


bool NetworkBuilderMPS::getParameter(const std::string & name, long long * value) const
{
 //`Finish
 return false;
}


bool NetworkBuilderMPS::setParameter(const std::string & name, long long value)
{
 //`Finish
 return false;
}


void NetworkBuilderMPS::build(TensorNetwork & network) const
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
