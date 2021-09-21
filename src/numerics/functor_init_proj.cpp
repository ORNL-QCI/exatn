/** ExaTN::Numerics: Tensor Functor: Initialization of Ordering Projection tensors
REVISION: 2021/09/21

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "functor_init_proj.hpp"

#include "tensor_range.hpp"

#include "talshxx.hpp"

namespace exatn{

namespace numerics{

void FunctorInitProj::pack(BytePacket & packet)
{
 return;
}


void FunctorInitProj::unpack(BytePacket & packet)
{
 return;
}


int FunctorInitProj::apply(talsh::Tensor & local_tensor) //tensor slice (in general)
{
 unsigned int rank;
 const auto * extents = local_tensor.getDimExtents(rank); //rank is returned by reference
 const auto tensor_volume = local_tensor.getVolume(); //volume of the given tensor slice
 const auto & offsets = local_tensor.getDimOffsets(); //base offsets of the given tensor slice

 assert(rank%2 == 0); //only even-order tensors

 auto init_proj_strong = [&](auto * tensor_body){
  std::vector<DimOffset> bas(rank);
  for(unsigned int i = 0; i < rank; ++i) bas[i] = offsets[i]; //tensor slice dimension base offsets
  std::vector<DimExtent> ext(rank);
  for(unsigned int i = 0; i < rank; ++i) ext[i] = extents[i]; //tensor slice dimension extents
  TensorRange rng(bas,ext);
  bool not_over = true;
  while(not_over){
   if(rng.increasingOrderDiag()){
    tensor_body[rng.localOffset()] = 1.0;
   }else{
    tensor_body[rng.localOffset()] = 0.0;
   }
   not_over = rng.next();
  }
  return 0;
 };

 auto init_proj_weak = [&](auto * tensor_body){
  std::vector<DimOffset> bas(rank);
  for(unsigned int i = 0; i < rank; ++i) bas[i] = offsets[i]; //tensor slice dimension base offsets
  std::vector<DimExtent> ext(rank);
  for(unsigned int i = 0; i < rank; ++i) ext[i] = extents[i]; //tensor slice dimension extents
  TensorRange rng(bas,ext);
  bool not_over = true;
  while(not_over){
   if(rng.nondecreasingOrderDiag()){
    tensor_body[rng.localOffset()] = 1.0;
   }else{
    tensor_body[rng.localOffset()] = 0.0;
   }
   not_over = rng.next();
  }
  return 0;
 };

 auto access_granted = false;
 {//Try REAL32:
  float * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted){
   if(weak_ordering_){
    return init_proj_weak(body);
   }else{
    return init_proj_strong(body);
   }
  }
 }

 {//Try REAL64:
  double * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted){
   if(weak_ordering_){
    return init_proj_weak(body);
   }else{
    return init_proj_strong(body);
   }
  }
 }

 {//Try COMPLEX32:
  std::complex<float> * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted){
   if(weak_ordering_){
    return init_proj_weak(body);
   }else{
    return init_proj_strong(body);
   }
  }
 }

 {//Try COMPLEX64:
  std::complex<double> * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted){
   if(weak_ordering_){
    return init_proj_weak(body);
   }else{
    return init_proj_strong(body);
   }
  }
 }

 std::cout << "#ERROR(exatn::numerics::FunctorInitProj): Unknown data kind in talsh::Tensor!" << std::endl;
 return 1;
}

} //namespace numerics

} //namespace exatn
