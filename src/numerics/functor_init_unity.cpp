/** ExaTN::Numerics: Tensor Functor: Initialization of an isometric tensor to unity
REVISION: 2022/06/14

Copyright (C) 2018-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle)
Copyright (C) 2022-2022 NVIDIA Corp. **/

#include "functor_init_unity.hpp"

#include "tensor_range.hpp"

#include "talshxx.hpp"

#include <type_traits>

namespace exatn{

namespace numerics{

void FunctorInitUnity::pack(BytePacket & packet)
{
 fatal_error("#FATAL(FunctorInitUnity::pack): Not implemented!");
 return;
}


void FunctorInitUnity::unpack(BytePacket & packet)
{
 fatal_error("#FATAL(FunctorInitUnity::unpack): Not implemented!");
 return;
}


int FunctorInitUnity::apply(talsh::Tensor & local_tensor) //tensor slice (in general)
{
 unsigned int rank;
 const auto * extents = local_tensor.getDimExtents(rank); //rank is returned by reference
 const auto tensor_volume = local_tensor.getVolume(); //volume of the given tensor slice
 const auto & offsets = local_tensor.getDimOffsets(); //base offsets of the given tensor slice

 /*//DEBUG:
 std::cout << "Initializing a local isometric tensor to unity:\n";
 std::cout << "Bases:";
 for(unsigned int i = 0; i < rank; ++i) std::cout << " " << offsets[i];
 std::cout << "; Extents:";
 for(unsigned int i = 0; i < rank; ++i) std::cout << " " << extents[i];
 std::cout << "; Iso-dims:";
 for(const auto & iso_dim: iso_dims_) std::cout << " " << iso_dim;
 std::cout << std::endl;*/

 const unsigned int iso_rank = iso_dims_.size();
 make_sure(iso_rank <= rank,
  "#ERROR(FunctorInitUnity::apply): Invalid specification of isometric dimensions!");
 const unsigned int other_rank = rank - iso_rank;

 std::vector<unsigned int> other_dims(other_rank);
 if(rank > 0){
  std::vector<unsigned int> marks(rank,0);
  for(const auto & iso_dim: iso_dims_){
   make_sure(iso_dim < rank,
    "#ERROR(FunctorInitUnity::apply): Invalid specification of isometric dimensions!");
   if(marks[iso_dim] == 0){
    marks[iso_dim] = 1;
    make_sure(offsets[iso_dim] == 0,
     "#ERROR(FunctorInitUnity::apply): Isometric tensor dimensions are not expected to have non-zero offsets!");
   }else{
    fatal_error("#ERROR(FunctorInitUnity::apply): Invalid specification of isometric dimensions!");
   }
  }
  unsigned int j = 0;
  for(unsigned int i = 0; i < rank; ++i){
   if(marks[i] == 0) other_dims[j++] = i;
  }
  assert(j == other_rank);
 }

 auto init_unity = [&](auto * tensor_body){

  std::vector<DimOffset> bas(rank);
  for(unsigned int i = 0; i < rank; ++i) bas[i] = offsets[i]; //tensor slice dimension base offsets
  std::vector<DimExtent> ext(rank);
  for(unsigned int i = 0; i < rank; ++i) ext[i] = extents[i]; //tensor slice dimension extents
  TensorRange rng(bas,ext);

  ext.resize(iso_rank);
  for(unsigned int i = 0; i < iso_rank; ++i) ext[i] = extents[iso_dims_[i]];
  TensorRange iso_rng(ext);

  bas.resize(other_rank);
  for(unsigned int i = 0; i < other_rank; ++i) bas[i] = offsets[other_dims[i]];
  ext.resize(other_rank);
  for(unsigned int i = 0; i < other_rank; ++i) ext[i] = extents[other_dims[i]];
  TensorRange other_rng(bas,ext);
  make_sure(iso_rng.localVolume() >= other_rng.localVolume(),
   "#ERROR(FunctorInitUnity::apply): The volume of the isometric dimension group must be no less than the rest!");

  const typename std::remove_pointer<decltype(tensor_body)>::type zero{0.0};
  const typename std::remove_pointer<decltype(tensor_body)>::type one{1.0};

  for(DimExtent v = 0; v < tensor_volume; ++v){
   tensor_body[v] = zero;
  }

  std::vector<DimOffset> mlndx(rank);
  const auto vol = other_rng.localVolume();
  for(DimExtent v = 0; v < vol; ++v){
   other_rng.reset(v);
   const auto & other_mlndx = other_rng.getMultiIndex();
   for(unsigned int i = 0; i < other_rank; ++i) mlndx[other_dims[i]] = other_mlndx[i];
   iso_rng.reset(v);
   const auto & iso_mlndx = iso_rng.getMultiIndex();
   for(unsigned int i = 0; i < iso_rank; ++i) mlndx[iso_dims_[i]] = iso_mlndx[i];
   rng.reset(mlndx);
   tensor_body[rng.localOffset()] = one;
   //for(const auto & ind: rng.getMultiIndex()) std::cout << " " << ind; //debug
   //std::cout << " = 1.0: Offset = " << rng.localOffset() << std::endl; //debug
  }

  return 0;
 };

 auto access_granted = false;
 {//Try REAL32:
  float * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted) return init_unity(body);
 }

 {//Try REAL64:
  double * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted) return init_unity(body);
 }

 {//Try COMPLEX32:
  std::complex<float> * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted) return init_unity(body);
 }

 {//Try COMPLEX64:
  std::complex<double> * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted) return init_unity(body);
 }

 std::cout << "#ERROR(exatn::numerics::FunctorInitUnity): Unknown data kind in talsh::Tensor!" << std::endl;
 return 1;
}

} //namespace numerics

} //namespace exatn
