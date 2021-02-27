/** ExaTN::Numerics: Composite tensor
REVISION: 2021/02/27

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_composite.hpp"
#include "space_register.hpp"

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
  subtensor.second->pack(byte_packet); //works correctly for both base Tensor and its derived classes
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
 auto space_reg = getSpaceRegister();
 //`Finish
 return;
}

} //namespace numerics

} //namespace exatn
