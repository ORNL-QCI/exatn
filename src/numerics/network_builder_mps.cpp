/** ExaTN::Numerics: Tensor network builder: MPS: Matrix Product State
REVISION: 2019/11/01

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
 auto output_tensor = network.getTensor(0);
 const auto output_tensor_rank = output_tensor->getRank();
 const auto & output_dim_extents = output_tensor->getDimExtents();
 DimExtent left_dim = 1;
 for(unsigned int i = 1; i <= output_tensor_rank/2; ++i){
  DimExtent right_dim = left_dim * output_dim_extents[i];
  if(i > 1){
   network.appendTensor(i,
                        std::make_shared<Tensor>("T"+std::to_string(i),
                                                 std::initializer_list<DimExtent>{left_dim,output_dim_extents[i],right_dim}
                                                ),
                        {TensorLeg{i-1,2},TensorLeg{0,i-1},TensorLeg{i+1,0}}
                       );
  }else{
   
  }
  left_dim = right_dim;
 }
 return;
}


std::unique_ptr<NetworkBuilder> NetworkBuilderMPS::createNew()
{
 return std::unique_ptr<NetworkBuilder>(new NetworkBuilderMPS());
}

} //namespace numerics

} //namespace exatn
