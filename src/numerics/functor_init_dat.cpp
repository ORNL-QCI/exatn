/** ExaTN::Numerics: Tensor Functor: Initialization to a given external data
REVISION: 2020/07/27

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "functor_init_dat.hpp"

#include "tensor_range.hpp"

#include "talshxx.hpp"

namespace exatn{

namespace numerics{

void FunctorInitDat::pack(BytePacket & packet)
{
 unsigned int rank = shape_.getRank();
 appendToBytePacket(&packet,rank);
 const auto & extents = shape_.getDimExtents();
 for(const auto & extent: extents) appendToBytePacket(&packet,extent);
 for(const auto & datum: data_) appendToBytePacket(&packet,datum);
 return;
}


void FunctorInitDat::unpack(BytePacket & packet)
{
 unsigned int rank;
 extractFromBytePacket(&packet,rank);
 std::vector<DimExtent> extents(rank);
 for(unsigned int i = 0; i < rank; ++i) extractFromBytePacket(&packet,extents[i]);
 shape_ = TensorShape(extents);
 data_.resize(shape_.getVolume());
 for(auto & datum: data_) extractFromBytePacket(&packet,datum);
 return;
}


int FunctorInitDat::apply(talsh::Tensor & local_tensor) //tensor slice (in general)
{
 unsigned int rank;
 const auto * extents = local_tensor.getDimExtents(rank); //rank is returned by reference
 assert(rank == shape_.getRank());
 const auto tensor_volume = local_tensor.getVolume(); //volume of the given tensor slice
 const auto & offsets = local_tensor.getDimOffsets(); //base offsets of the given tensor slice
 DimExtent full_volume; //volume of the full tensor
 auto full_strides = shape_.getDimStrides(&full_volume); //full tensor strides
 assert(full_strides.size() == rank);
 assert(tensor_volume <= full_volume);
 for(unsigned int i = 0; i < rank; ++i){
  if(offsets[i] + extents[i] > shape_.getDimExtent(i)){
   std::cout << "#ERROR(exatn::numerics::FunctorInitDat): Tensor dimension mismatch for dimension "
             << i << ": " << offsets[i] << " " << extents[i] << " " << shape_.getDimExtent(i) << std::endl;
   local_tensor.print();
   assert(false);
   return 2;
  }
 }

 std::vector<DimOffset> bas(rank);
 for(unsigned int i = 0; i < rank; ++i) bas[i] = offsets[i]; //tensor slice dimension base offsets
 std::vector<DimExtent> ext(rank);
 for(unsigned int i = 0; i < rank; ++i) ext[i] = extents[i]; //tensor slice dimension extents
 TensorRange rng(bas,ext,full_strides);

 auto access_granted = false;
 {//Try REAL32:
  float * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted){
   bool more = true;
   while(more){
    body[rng.localOffset()] = static_cast<float>(data_[rng.globalOffset()].real());
    more = rng.next();
   }
   return 0;
  }
 }

 {//Try REAL64:
  double * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted){
   bool more = true;
   while(more){
    body[rng.localOffset()] = data_[rng.globalOffset()].real();
    more = rng.next();
   }
   return 0;
  }
 }

 {//Try COMPLEX32:
  std::complex<float> * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted){
   bool more = true;
   while(more){
    body[rng.localOffset()] = std::complex<float>(data_[rng.globalOffset()]);
    more = rng.next();
   }
   return 0;
  }
 }

 {//Try COMPLEX64:
  std::complex<double> * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted){
   bool more = true;
   while(more){
    body[rng.localOffset()] = data_[rng.globalOffset()];
    more = rng.next();
   }
   return 0;
  }
 }

 std::cout << "#ERROR(exatn::numerics::FunctorInitDat): Unknown data kind in talsh::Tensor!" << std::endl;
 return 1;
}

} //namespace numerics

} //namespace exatn
