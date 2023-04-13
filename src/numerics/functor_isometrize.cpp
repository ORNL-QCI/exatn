/** ExaTN::Numerics: Tensor Functor: Tensor Isometrization
REVISION: 2022/09/13

Copyright (C) 2018-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle)
Copyright (C) 2022-2022 NVIDIA Corporation

SPDX-License-Identifier: BSD-3-Clause **/

#include "functor_isometrize.hpp"

#include "tensor_range.hpp"

#include "talshxx.hpp"

#include <random>
#include <functional>

namespace exatn{

namespace numerics{

void FunctorIsometrize::pack(BytePacket & packet)
{
 fatal_error("#FATAL(FunctorIsometrize::pack): Not implemented!");
 return;
}


void FunctorIsometrize::unpack(BytePacket & packet)
{
 fatal_error("#FATAL(FunctorIsometrize::unpack): Not implemented!");
 return;
}


// Complex conjugation:
template <typename NumericType>
inline NumericType conjugated(NumericType value)
{
 return value;
}

template <typename NumericType>
inline std::complex<NumericType> conjugated(std::complex<NumericType> value)
{
 return std::conj(value);
}


// MGS for float and double:
template<typename NumericType>
void modifiedGramSchmidt(NumericType * tensor_body, TensorRange & rngx, TensorRange & rngy)
{
 //Allocate a tempory matricized buffer:
 DimExtent volx = rngx.localVolume();
 DimExtent voly = rngy.localVolume();
 auto * buf = new double[voly*volx];

 //Load data into the tempory matricized buffer:
 rngy.reset();
#pragma omp parallel for schedule(guided) shared(rngx,rngy,volx,voly,buf,tensor_body)
 for(DimOffset j = 0; j < voly; ++j){
  rngx.reset();
  for(DimOffset i = 0; i < volx; ++i){
   buf[volx*j + i] = static_cast<double>(tensor_body[rngy.globalOffset() + rngx.globalOffset()]);
   rngx.next();
  }
  rngy.next();
 }
#if 0
 //Print input matrix (debug):
 std::cout << "MGS input:\n" << std::scientific;
 for(DimOffset i = 0; i < volx; ++i){
  for(DimOffset j = 0; j < voly; ++j){
   std::cout << "  " << buf[volx*j + i];
  }
  std::cout << std::endl;
 }
#endif
 //Modified Gram-Schmidt procedure:
 for(DimOffset j = 0; j < voly; ++j){
  double nrm2 {0.0};
#pragma omp parallel for schedule(guided) shared(j,volx,buf) reduction(+:nrm2)
  for(DimOffset i = 0; i < volx; ++i){
   const double elem = std::abs(buf[volx*j + i]);
   nrm2 += elem * elem;
  }
  nrm2 = std::sqrt(nrm2);
#pragma omp parallel for schedule(guided) shared(j,volx,buf,nrm2)
  for(DimOffset i = 0; i < volx; ++i){
   buf[volx*j + i] /= nrm2;
  }
#pragma omp parallel for schedule(guided) shared(j,volx,voly,buf)
  for(DimOffset k = j+1; k < voly; ++k){
   double dpr {0.0};
   for(DimOffset i = 0; i < volx; ++i){
    dpr += conjugated(buf[volx*j + i]) * buf[volx*k + i];
   }
   for(DimOffset i = 0; i < volx; ++i){
    buf[volx*k + i] -= dpr * buf[volx*j + i];
   }
  }
 }
#if 0
 //Print output matrix (debug):
 std::cout << "MGS output:\n" << std::scientific;
 for(DimOffset i = 0; i < volx; ++i){
  for(DimOffset j = 0; j < voly; ++j){
   std::cout << "  " << buf[volx*j + i];
  }
  std::cout << std::endl;
 }
#endif
#if 1
 //Verification (debug):
 for(DimOffset j = 0; j < voly; ++j){
  double nrm2 {0.0};
  for(DimOffset k = 0; k < volx; ++k){
   const double elem = std::abs(buf[volx*j + k]);
   nrm2 += elem * elem;
  }
  nrm2 = std::sqrt(nrm2);
  make_sure(nrm2,1.0,1e-5,
   "#FATAL(FunctorIsometrize::apply): MGS procedure failed in norm: " + std::to_string(nrm2));
  for(DimOffset i = (j+1); i < voly; ++i){
   double overlap {0.0};
   for(DimOffset k = 0; k < volx; ++k){
    overlap += conjugated(buf[volx*j + k]) * buf[volx*i + k];
   }
   make_sure(static_cast<double>(std::abs(overlap)),0.0,1e-5,
    "#FATAL(FunctorIsometrize::apply): MGS procedure failed in overlap: " + std::to_string(std::abs(overlap)));
  }
 }
#endif
 //Copy the result back into the tensor:
 rngy.reset();
#pragma omp parallel for schedule(guided) shared(rngx,rngy,volx,voly,buf,tensor_body)
 for(DimOffset j = 0; j < voly; ++j){
  rngx.reset();
  for(DimOffset i = 0; i < volx; ++i){
   tensor_body[rngy.globalOffset() + rngx.globalOffset()] = static_cast<NumericType>(buf[volx*j + i]);
   rngx.next();
  }
  rngy.next();
 }

 //Deallocate the temporary matricized buffer:
 delete [] buf;
 return;
}


// MGS for complex float and double:
template<typename NumericType>
void modifiedGramSchmidt(std::complex<NumericType> * tensor_body, TensorRange & rngx, TensorRange & rngy)
{
 //Allocate a tempory matricized buffer:
 DimExtent volx = rngx.localVolume();
 DimExtent voly = rngy.localVolume();
 auto * buf = new std::complex<double>[voly*volx];

 //Load data into the tempory matricized buffer:
 rngy.reset();
#pragma omp parallel for schedule(guided) shared(rngx,rngy,volx,voly,buf,tensor_body)
 for(DimOffset j = 0; j < voly; ++j){
  rngx.reset();
  for(DimOffset i = 0; i < volx; ++i){
   buf[volx*j + i] = std::complex<double>(tensor_body[rngy.globalOffset() + rngx.globalOffset()]);
   rngx.next();
  }
  rngy.next();
 }
#if 0
 //Print input matrix (debug):
 std::cout << "MGS input:\n" << std::scientific;
 for(DimOffset i = 0; i < volx; ++i){
  for(DimOffset j = 0; j < voly; ++j){
   std::cout << "  " << buf[volx*j + i];
  }
  std::cout << std::endl;
 }
#endif
 //Modified Gram-Schmidt procedure:
 for(DimOffset j = 0; j < voly; ++j){
  double nrm2 {0.0};
#pragma omp parallel for schedule(guided) shared(j,volx,buf) reduction(+:nrm2)
  for(DimOffset i = 0; i < volx; ++i){
   const double elem = std::abs(buf[volx*j + i]);
   nrm2 += elem * elem;
  }
  nrm2 = std::sqrt(nrm2);
#pragma omp parallel for schedule(guided) shared(j,volx,buf,nrm2)
  for(DimOffset i = 0; i < volx; ++i){
   buf[volx*j + i] /= nrm2;
  }
#pragma omp parallel for schedule(guided) shared(j,volx,voly,buf)
  for(DimOffset k = j+1; k < voly; ++k){
   std::complex<double> dpr {0.0,0.0};
   for(DimOffset i = 0; i < volx; ++i){
    dpr += conjugated(buf[volx*j + i]) * buf[volx*k + i];
   }
   for(DimOffset i = 0; i < volx; ++i){
    buf[volx*k + i] -= dpr * buf[volx*j + i];
   }
  }
 }
#if 0
 //Print output matrix (debug):
 std::cout << "MGS output:\n" << std::scientific;
 for(DimOffset i = 0; i < volx; ++i){
  for(DimOffset j = 0; j < voly; ++j){
   std::cout << "  " << buf[volx*j + i];
  }
  std::cout << std::endl;
 }
#endif
#if 1
 //Verification (debug):
 for(DimOffset j = 0; j < voly; ++j){
  double nrm2 {0.0};
  for(DimOffset k = 0; k < volx; ++k){
   const double elem = std::abs(buf[volx*j + k]);
   nrm2 += elem * elem;
  }
  nrm2 = std::sqrt(nrm2);
  make_sure(nrm2,1.0,1e-5,
   "#FATAL(FunctorIsometrize::apply): MGS procedure failed in norm: " + std::to_string(nrm2));
  for(DimOffset i = (j+1); i < voly; ++i){
   std::complex<double> overlap {0.0,0.0};
   for(DimOffset k = 0; k < volx; ++k){
    overlap += conjugated(buf[volx*j + k]) * buf[volx*i + k];
   }
   make_sure(static_cast<double>(std::abs(overlap)),0.0,1e-5,
    "#FATAL(FunctorIsometrize::apply): MGS procedure failed in overlap: " + std::to_string(std::abs(overlap)));
  }
 }
#endif
 //Copy the result back into the tensor:
 rngy.reset();
#pragma omp parallel for schedule(guided) shared(rngx,rngy,volx,voly,buf,tensor_body)
 for(DimOffset j = 0; j < voly; ++j){
  rngx.reset();
  for(DimOffset i = 0; i < volx; ++i){
   tensor_body[rngy.globalOffset() + rngx.globalOffset()] = std::complex<NumericType>(buf[volx*j + i]);
   rngx.next();
  }
  rngy.next();
 }

 //Deallocate the temporary matricized buffer:
 delete [] buf;
 return;
}


int FunctorIsometrize::apply(talsh::Tensor & local_tensor) //tensor slice (in general)
{
 unsigned int tens_rank;
 const auto * extents = local_tensor.getDimExtents(tens_rank); //rank is returned by reference
 const auto tensor_volume = local_tensor.getVolume(); //volume of the given tensor slice
 const auto & offsets = local_tensor.getDimOffsets(); //base offsets of the given tensor slice
 for(const auto & offs: offsets) assert(offs == 0); //`tensor slices are not allowed (replicated tensors only)
 std::vector<DimExtent> strides(tens_rank);
 DimExtent stride = 1;
 for(int i = 0; i < tens_rank; ++i){
  strides[i] = stride;
  stride *= extents[i];
 }

 /*std::random_device seeder;
 std::default_random_engine generator(seeder());
 std::uniform_real_distribution<float> distribution_float(-1.0,1.0);
 auto rnd_float = std::bind(distribution_float,generator);
 std::uniform_real_distribution<double> distribution_double(-1.0,1.0);
 auto rnd_double = std::bind(distribution_double,generator);*/

 auto enforce_isometry = [&](auto * tensor_body,
                             const std::vector<unsigned int> & iso_dims){
  using tensor_body_type = typename std::remove_pointer<decltype(tensor_body)>::type;

  unsigned int rankx = iso_dims.size();
  unsigned int ranky = tens_rank - rankx;
  if(rankx > 0){
   //Set up ranges:
   std::vector<DimExtent> extx(rankx), strx(rankx);
   std::vector<DimExtent> exty(ranky), stry(ranky);
   std::vector<int> dim_mask(tens_rank,1);
   for(const auto & dim: iso_dims){
    assert(dim >= 0 && dim < tens_rank);
    dim_mask[dim] = 0;
   }
   int x = 0, y = 0;
   for(int i = 0; i < tens_rank; ++i){
    if(dim_mask[i] == 0){
     extx[x] = extents[i];
     strx[x++] = strides[i];
    }else{
     exty[y] = extents[i];
     stry[y++] = strides[i];
    }
   }
   assert(x == rankx && y == ranky);
   if(ranky == 0){
    exty.emplace_back(1);
    stry.emplace_back(1);
    ranky = 1;
   }
   TensorRange rngx(std::vector<DimOffset>(rankx,0),extx,strx);
   TensorRange rngy(std::vector<DimOffset>(ranky,0),exty,stry);
   modifiedGramSchmidt(tensor_body,rngx,rngy);
  }
  return 0;
 };

 auto access_granted = false;
 {//Try REAL32:
  float * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted) return enforce_isometry(body,isometry1_);
 }

 {//Try REAL64:
  double * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted) return enforce_isometry(body,isometry1_);
 }

 {//Try COMPLEX32:
  std::complex<float> * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted) return enforce_isometry(body,isometry1_);
 }

 {//Try COMPLEX64:
  std::complex<double> * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted) return enforce_isometry(body,isometry1_);
 }

 std::cout << "#ERROR(exatn::numerics::FunctorIsometrize): Unknown data kind in talsh::Tensor!" << std::endl;
 return 1;
}

} //namespace numerics

} //namespace exatn
