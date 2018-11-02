/** ExaTN::Numerics: Tensor signature
REVISION: 2018/11/02

Copyright (C) 2018-2018 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2018 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_signature.hpp"

#include <iostream>
#include <iterator>

namespace exatn{

namespace numerics{

TensorSignature::TensorSignature(std::initializer_list<std::pair<SpaceId,SubspaceId>> subspaces):
subspaces_(subspaces)
{
}

TensorSignature::TensorSignature(const std::vector<std::pair<SpaceId,SubspaceId>> & subspaces):
subspaces_(subspaces)
{
}

TensorSignature::TensorSignature(unsigned int rank):
subspaces_(rank,std::pair<SpaceId,SubspaceId>(SOME_SPACE,0))
{
}

void TensorSignature::printIt() const
{
 std::cout << "{";
 for(auto subsp_it = subspaces_.cbegin(); subsp_it != subspaces_.cend(); ++subsp_it){
  if(std::next(subsp_it,1) == subspaces_.cend()){
   std::cout << std::get<0>(*subsp_it) << ":" << std::get<1>(*subsp_it);
  }else{
   std::cout << std::get<0>(*subsp_it) << ":" << std::get<1>(*subsp_it) << ",";
  }
 }
 std::cout << "}";
 return;
}

unsigned int TensorSignature::getRank() const
{
 return static_cast<unsigned int>(subspaces_.size());
}

SpaceId TensorSignature::getDimSpaceId(unsigned int dim_id) const
{
 return std::get<0>(subspaces_.at(dim_id));
}

SubspaceId TensorSignature::getDimSubspaceId(unsigned int dim_id) const
{
 return std::get<1>(subspaces_.at(dim_id));
}

std::pair<SpaceId,SubspaceId> TensorSignature::getDimSpaceAttr(unsigned int dim_id) const
{
 return subspaces_.at(dim_id);
}

} //namespace numerics

} //namespace exatn
