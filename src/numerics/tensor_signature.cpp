/** ExaTN::Numerics: Tensor signature
REVISION: 2020/06/25

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_signature.hpp"

#include <iterator>
#include <cassert>

namespace exatn{

namespace numerics{

TensorSignature::TensorSignature()
{
}

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

TensorSignature::TensorSignature(const TensorSignature & another,
                                 const std::vector<unsigned int> & order):
 TensorSignature(another)
{
 const auto rank = another.getRank();
 assert(order.size() == rank);
 const auto & orig = another.getDimSpaceAttrs();
 for(unsigned int new_pos = 0; new_pos < rank; ++new_pos) subspaces_[new_pos] = orig[order[new_pos]];
}

void TensorSignature::pack(BytePacket & byte_packet) const
{
 const std::size_t tensor_rank = subspaces_.size();
 appendToBytePacket(&byte_packet,tensor_rank);
 for(const auto & subspace: subspaces_) appendToBytePacket(&byte_packet,subspace);
 return;
}

void TensorSignature::unpack(BytePacket & byte_packet)
{
 std::size_t tensor_rank = 0;
 extractFromBytePacket(&byte_packet,tensor_rank);
 subspaces_.resize(tensor_rank);
 for(auto & subspace: subspaces_) extractFromBytePacket(&byte_packet,subspace);
 return;
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

const std::vector<std::pair<SpaceId,SubspaceId>> & TensorSignature::getDimSpaceAttrs() const
{
 return subspaces_;
}

bool TensorSignature::isCongruentTo(const TensorSignature & another) const
{
 const auto rank = this->getRank();
 if(another.getRank() != rank) return false;
 for(unsigned int i = 0; i < rank; ++i){
  if(this->getDimSpaceAttr(i) != another.getDimSpaceAttr(i)) return false;
 }
 return true;
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
