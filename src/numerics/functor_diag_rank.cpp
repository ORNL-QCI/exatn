/** ExaTN::Numerics: Tensor Functor: Computes partial 2-norms over a given tensor dimension
REVISION: 2020/04/23

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "functor_diag_rank.hpp"

#include "tensor_range.hpp"

#include "talshxx.hpp"

namespace exatn{

namespace numerics{

FunctorDiagRank::FunctorDiagRank(unsigned int tensor_dimension):
 tensor_dimension_(tensor_dimension)
{
}

void FunctorDiagRank::pack(BytePacket & packet)
{
 std::size_t extent = partial_norms_.size();
 appendToBytePacket(&packet,tensor_dimension_);
 appendToBytePacket(&packet,extent);
 for(const auto & norm: partial_norms_) appendToBytePacket(&packet,norm);
 return;
}


void FunctorDiagRank::unpack(BytePacket & packet)
{
 std::size_t extent;
 extractFromBytePacket(&packet,tensor_dimension_);
 extractFromBytePacket(&packet,extent);
 partial_norms_.resize(extent);
 for(auto & norm: partial_norms_) extractFromBytePacket(&packet,norm);
 return;
}


int FunctorDiagRank::apply(talsh::Tensor & local_tensor) //tensor slice (in general)
{
 unsigned int rank;
 const auto * extents = local_tensor.getDimExtents(rank); //rank is returned by reference
 const auto tensor_volume = local_tensor.getVolume(); //volume of the given tensor slice

 return 0;
}

} //namespace numerics

} //namespace exatn
