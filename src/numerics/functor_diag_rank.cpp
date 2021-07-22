/** ExaTN::Numerics: Tensor Functor: Computes partial 2-norms over a given tensor dimension
REVISION: 2021/07/22

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "functor_diag_rank.hpp"

#include "tensor_range.hpp"

#include "talshxx.hpp"

namespace exatn{

namespace numerics{

std::mutex FunctorDiagRank::mutex_;

FunctorDiagRank::FunctorDiagRank(unsigned int tensor_dimension, DimExtent dimension_extent):
 tensor_dimension_(tensor_dimension), partial_norms_(dimension_extent,0.0)
{
}


void FunctorDiagRank::pack(BytePacket & packet)
{
 const std::lock_guard<std::mutex> lock(mutex_);
 std::size_t extent = partial_norms_.size();
 appendToBytePacket(&packet,tensor_dimension_);
 appendToBytePacket(&packet,extent);
 for(const auto & norm: partial_norms_) appendToBytePacket(&packet,norm);
 return;
}


void FunctorDiagRank::unpack(BytePacket & packet)
{
 const std::lock_guard<std::mutex> lock(mutex_);
 std::size_t extent;
 extractFromBytePacket(&packet,tensor_dimension_);
 extractFromBytePacket(&packet,extent);
 partial_norms_.resize(extent);
 for(auto & norm: partial_norms_) extractFromBytePacket(&packet,norm);
 return;
}


int FunctorDiagRank::apply(talsh::Tensor & local_tensor) //tensor slice (in general)
{
 const std::lock_guard<std::mutex> lock(mutex_);
 unsigned int tensor_rank;
 const auto * tensor_dims = local_tensor.getDimExtents(tensor_rank); //rank is returned by reference
 const auto & tensor_base = local_tensor.getDimOffsets(); //dimension base offsets
 const auto tensor_volume = local_tensor.getVolume(); //volume of the given tensor
 std::vector<DimOffset> bases(tensor_rank);
 for(unsigned int i = 0; i < tensor_rank; ++i) bases[i] = tensor_base[i];
 std::vector<DimExtent> extents(tensor_rank);
 for(unsigned int i = 0; i < tensor_rank; ++i) extents[i] = tensor_dims[i];

 assert(tensor_dimension_ < tensor_rank);
 const auto upper = bases[tensor_dimension_] + extents[tensor_dimension_];
 assert(upper <= partial_norms_.size());

 auto procedure = [&](const auto * tens_body){ //`Needs multi-threading
  TensorRange body_range(bases,extents);
  bool active = (body_range.localVolume() > 0);
  while(active){
   const auto index_value = bases[tensor_dimension_] + body_range.getIndex(tensor_dimension_);
   const auto offset = body_range.localOffset();
   double val = std::abs(tens_body[offset]);
   partial_norms_[index_value] += val * val;
   active = body_range.next();
  }
  return 0;
 };

 { //Try float
  const float * tensor_body;
  bool access_granted = local_tensor.getDataAccessHostConst(&tensor_body);
  if(access_granted) return procedure(tensor_body);
 }

 { //Try double
  const double * tensor_body;
  bool access_granted = local_tensor.getDataAccessHostConst(&tensor_body);
  if(access_granted) return procedure(tensor_body);
 }

 { //Try double
  const std::complex<float> * tensor_body;
  bool access_granted = local_tensor.getDataAccessHostConst(&tensor_body);
  if(access_granted) return procedure(tensor_body);
 }

 { //Try double
  const std::complex<double> * tensor_body;
  bool access_granted = local_tensor.getDataAccessHostConst(&tensor_body);
  if(access_granted) return procedure(tensor_body);
 }

 std::cout << "#ERROR(exatn::numerics::FunctorDiagRank): Unknown data kind in talsh::Tensor!" << std::endl;
 return 1;
}

} //namespace numerics

} //namespace exatn
