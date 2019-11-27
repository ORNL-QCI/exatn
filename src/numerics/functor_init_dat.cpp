/** ExaTN::Numerics: Tensor Functor: Initialization to a given external data
REVISION: 2019/11/26

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "functor_init_dat.hpp"

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
 for(auto & datum: data_) extractFromBytePacket(&packet,datum);
 return;
}


int FunctorInitDat::apply(talsh::Tensor & local_tensor)
{
 unsigned int rank;
 const auto * extents = local_tensor.getDimExtents(rank);
 auto tensor_volume = local_tensor.getVolume();
 const auto & offsets = local_tensor.getDimOffsets();

 auto access_granted = false;

 {//Try REAL32:
  float * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted){

#pragma omp parallel for schedule(guided) shared(tensor_volume,body)
   for(std::size_t i = 0; i < tensor_volume; ++i) body[i] = 0.0f;
   return 0;
  }
 }

 {//Try REAL64:
  double * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted){

#pragma omp parallel for schedule(guided) shared(tensor_volume,body)
   for(std::size_t i = 0; i < tensor_volume; ++i) body[i] = 0.0;
   return 0;
  }
 }

 {//Try COMPLEX32:
  std::complex<float> * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted){

#pragma omp parallel for schedule(guided) shared(tensor_volume,body)
   for(std::size_t i = 0; i < tensor_volume; ++i) body[i] = std::complex<float>{0.0f,0.0f};
   return 0;
  }
 }

 {//Try COMPLEX64:
  std::complex<double> * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted){

#pragma omp parallel for schedule(guided) shared(tensor_volume,body)
   for(std::size_t i = 0; i < tensor_volume; ++i) body[i] = std::complex<double>{0.0,0.0};
   return 0;
  }
 }

 std::cout << "#ERROR(exatn::numerics::FunctorInitDat): Unknown data kind in talsh::Tensor!" << std::endl;
 return 1;
}

} //namespace numerics

} //namespace exatn
