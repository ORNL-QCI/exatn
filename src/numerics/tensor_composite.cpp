/** ExaTN::Numerics: Composite tensor
REVISION: 2021/03/05

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_composite.hpp"
#include "space_register.hpp"

#include <algorithm>

namespace exatn{

namespace numerics{

TensorComposite::TensorComposite(BytePacket & byte_packet):
 Tensor(byte_packet)
{
 unpackTensorComposite(byte_packet);
}


void TensorComposite::packTensorComposite(BytePacket & byte_packet) const
{
 //Member split_dims_:
 unsigned int n = split_dims_.size();
 appendToBytePacket(&byte_packet,n);
 for(unsigned int i = 0; i < n; ++i){
  appendToBytePacket(&byte_packet,split_dims_[i].first);
  appendToBytePacket(&byte_packet,split_dims_[i].second);
 }
 //Member dim_depth_:
 n = dim_depth_.size();
 appendToBytePacket(&byte_packet,n);
 for(unsigned int i = 0; i < n; ++i){
  appendToBytePacket(&byte_packet,dim_depth_[i]);
 }
 //Member num_bisections_ & bisect_bits_:
 appendToBytePacket(&byte_packet,num_bisections_);
 for(unsigned int i = 0; i < num_bisections_; ++i){
  appendToBytePacket(&byte_packet,bisect_bits_[i].first);
  appendToBytePacket(&byte_packet,bisect_bits_[i].second);
 }
 //Member subtensors_:
 unsigned long long num_subtensors = subtensors_.size();
 appendToBytePacket(&byte_packet,num_subtensors);
 for(const auto & subtensor: subtensors_){
  appendToBytePacket(&byte_packet,subtensor.first);
  subtensor.second->pack(byte_packet); //works correctly for both base Tensor and its derived classes (unpack does not, see below)
 }
 return;
}


void TensorComposite::unpackTensorComposite(BytePacket & byte_packet)
{
 //Member split_dims_:
 unsigned int n;
 extractFromBytePacket(&byte_packet,n);
 split_dims_.resize(n);
 for(unsigned int i = 0; i < n; ++i){
  extractFromBytePacket(&byte_packet,split_dims_[i].first);
  extractFromBytePacket(&byte_packet,split_dims_[i].second);
 }
 //Member dim_depth_:
 extractFromBytePacket(&byte_packet,n);
 dim_depth_.resize(n);
 for(unsigned int i = 0; i < n; ++i){
  extractFromBytePacket(&byte_packet,dim_depth_[i]);
 }
 //Member num_bisections_ & bisect_bits_:
 extractFromBytePacket(&byte_packet,num_bisections_);
 bisect_bits_.resize(num_bisections_);
 for(unsigned int i = 0; i < num_bisections_; ++i){
  extractFromBytePacket(&byte_packet,bisect_bits_[i].first);
  extractFromBytePacket(&byte_packet,bisect_bits_[i].second);
 }
 //Member subtensors_:
 subtensors_.clear();
 unsigned long long num_subtensors;
 extractFromBytePacket(&byte_packet,num_subtensors);
 for(unsigned long long i = 0; i < num_subtensors; ++i){
  unsigned long long key;
  extractFromBytePacket(&byte_packet,key);
  auto res = subtensors_.emplace(std::make_pair(key,std::make_shared<Tensor>(byte_packet))); //`only unpacks base Tensor
  assert(res.second);
 }
 return;
}


void TensorComposite::pack(BytePacket & byte_packet) const
{
 Tensor::pack(byte_packet);
 packTensorComposite(byte_packet);
 return;
}


void TensorComposite::unpack(BytePacket & byte_packet)
{
 Tensor::unpack(byte_packet);
 unpackTensorComposite(byte_packet);
 return;
}


void TensorComposite::generateSubtensors(std::function<bool (const Tensor &)> tensor_predicate)
{
 subtensors_.clear();
 auto space_reg = getSpaceRegister();
 const auto tensor_rank = getRank();
 if(tensor_rank > 0){
  //Split given subspaces to the requested depth:
  std::vector<std::pair<const Subspace *,                      //parental subspace
                        std::vector<std::shared_ptr<Subspace>> //child subspaces
                       >> subspaces(tensor_rank);
  for(unsigned int i = 0; i < tensor_rank; ++i){
   //Associate parental subspace:
   const auto subspace_attr = this->getDimSpaceAttr(i);
   if(subspace_attr.first == SOME_SPACE){
    const auto * some_space = space_reg->getSpace(SOME_SPACE);
    subspaces[i].first = new Subspace(some_space,0,some_space->getDimension()-1); //owning
   }else{
    subspaces[i].first = space_reg->getSubspace(subspace_attr.first,subspace_attr.second); //non-owning
   }
   assert(subspaces[i].first != nullptr);
   //Generate child subspaces:
   DimExtent dim_depth = getDimDepth(i);
   if(dim_depth > 0){
    const DimExtent num_subspaces = std::pow(2,dim_depth);
    assert(subspaces[i].first->getDimension() >= num_subspaces);
    subspaces[i].second = subspaces[i].first->splitUniform(num_subspaces);
    if(subspace_attr.first != SOME_SPACE){ //register named child subspaces
     for(auto subspace: subspaces[i].second) space_reg->registerSubspace(subspace);
    }
   }
  }
  //Iterate over all child subspaces and generate subtensors:
  std::vector<SubspaceId> subspace_signature(tensor_rank);
  std::vector<DimExtent> dim_extents(tensor_rank);
  std::vector<unsigned long long> subspace_id(tensor_rank,0ULL);
  const unsigned long long num_subtensors = getNumSubtensorsComplete();
  for(unsigned long long subtensor_id = 0; subtensor_id < num_subtensors; ++subtensor_id){
   std::fill(subspace_id.begin(),subspace_id.end(),0ULL);
   //Pick necessary subspaces:
   for(unsigned int i = 0; i < num_bisections_; ++i){ //bit position in bisect_bits_
    unsigned int bit_pos = ((num_bisections_ - 1) - i); //bit position in subtensor_id
    const auto bit = subtensor_id & (1ULL << bit_pos);
    const auto dim = bisect_bits_[i].first;
    subspace_id[dim] <<= 1;
    if(bit != 0) subspace_id[dim]++;
   }
   //Create and store the corresponding subtensor:
   for(unsigned int i = 0; i < tensor_rank; ++i){
    const auto subspace_attr = this->getDimSpaceAttr(i);
    if(getDimDepth(i) > 0){
     const auto & child_subspace = *(subspaces[i].second[subspace_id[i]]);
     if(subspace_attr.first == SOME_SPACE){
      subspace_signature[i] = child_subspace.getLowerBound();
      dim_extents[i] = child_subspace.getDimension();
     }else{
      subspace_signature[i] = child_subspace.getRegisteredId();
      dim_extents[i] = child_subspace.getDimension();
     }
    }else{
     subspace_signature[i] = subspace_attr.second;
     dim_extents[i] = getDimExtent(i);
    }
   }
   auto subtensor = createSubtensor(subspace_signature,dim_extents);
   if(tensor_predicate(*subtensor)){
    auto res = subtensors_.emplace(std::make_pair(subtensor_id,subtensor)); assert(res.second);
   }
  }
  //Clean:
  for(unsigned int i = 0; i < tensor_rank; ++i){
   if(this->getDimSpaceAttr(i).first == SOME_SPACE) delete subspaces[i].first;
   subspaces[i].first = nullptr;
  }
 }else{ //scalar tensor: Register itself as a subtensor
  auto res = subtensors_.emplace(std::make_pair(0ULL,createSubtensor({},{}))); assert(res.second);
 }
 return;
}

} //namespace numerics

} //namespace exatn
