/** ExaTN::Numerics: Tensor signature
REVISION: 2019/10/13

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_signature.hpp"

#include <iterator>
#include <cassert>

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

TensorSignature::TensorSignature()
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

void TensorSignature::printItFile(std::ofstream & output_file) const
{
 output_file << "{";
 for(auto subsp_it = subspaces_.cbegin(); subsp_it != subspaces_.cend(); ++subsp_it){
  if(std::next(subsp_it,1) == subspaces_.cend()){
   output_file << std::get<0>(*subsp_it) << ":" << std::get<1>(*subsp_it);
  }else{
   output_file << std::get<0>(*subsp_it) << ":" << std::get<1>(*subsp_it) << ",";
  }
 }
 output_file << "}";
 return;
}

unsigned int TensorSignature::getRank() const
{
 return static_cast<unsigned int>(subspaces_.size());
}

SpaceId TensorSignature::getDimSpaceId(unsigned int dim_id) const
{
 assert(dim_id < subspaces_.size()); //debug
 return std::get<0>(subspaces_[dim_id]);
}

SubspaceId TensorSignature::getDimSubspaceId(unsigned int dim_id) const
{
 assert(dim_id < subspaces_.size()); //debug
 return std::get<1>(subspaces_[dim_id]);
}

std::pair<SpaceId,SubspaceId> TensorSignature::getDimSpaceAttr(unsigned int dim_id) const
{
 assert(dim_id < subspaces_.size()); //debug
 return subspaces_[dim_id];
}

void TensorSignature::resetDimension(unsigned int dim_id, std::pair<SpaceId,SubspaceId> subspace)
{
 assert(dim_id < subspaces_.size()); //debug
 subspaces_[dim_id] = subspace;
 return;
}

void TensorSignature::deleteDimension(unsigned int dim_id)
{
 assert(dim_id < subspaces_.size());
 subspaces_.erase(subspaces_.cbegin()+dim_id);
 return;
}

void TensorSignature::appendDimension(std::pair<SpaceId,SubspaceId> subspace)
{
 subspaces_.emplace_back(subspace);
 return;
}

} //namespace numerics

} //namespace exatn
