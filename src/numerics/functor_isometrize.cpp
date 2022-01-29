/** ExaTN::Numerics: Tensor Functor: Tensor Isometrization
REVISION: 2022/01/29

Copyright (C) 2018-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle) **/

#include "functor_isometrize.hpp"

#include "tensor_range.hpp"

#include "talshxx.hpp"

namespace exatn{

namespace numerics{

void FunctorIsometrize::pack(BytePacket & packet)
{
 //`Finish
 return;
}


void FunctorIsometrize::unpack(BytePacket & packet)
{
 //`Finish
 return;
}


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

 auto enforce_isometry = [&](auto * tensor_body,
                             const std::vector<unsigned int> & iso_dims){
  using tensor_body_type = typename std::remove_pointer<decltype(tensor_body)>::type;

  unsigned int rankx = iso_dims.size();
  unsigned int ranky = tens_rank - rankx;
  if(rankx > 0 && ranky > 0){
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
   TensorRange rngx(std::vector<DimOffset>(rankx,0),extx,strx);
   TensorRange rngy(std::vector<DimOffset>(ranky,0),exty,stry);
   DimExtent volx = rngx.localVolume();
   DimExtent voly = rngy.localVolume();

   //Allocate a tempory matricized buffer:
   tensor_body_type * buf = new tensor_body_type[voly*volx];

   //Load data into the tempory matricized buffer:
   rngy.reset();
   for(DimOffset j = 0; j < voly; ++j){
    rngx.reset();
    for(DimOffset i = 0; i < volx; ++i){
     buf[volx*j + i] = tensor_body[rngy.globalOffset() + rngx.globalOffset()];
     rngx.next();
    }
    rngy.next();
   }

   //Modified Gram-Schmidt procedure:
   for(DimOffset j = 0; j < voly; ++j){
    double nrm2 = 0.0;
    for(DimOffset i = 0; i < volx; ++i){
     const auto elem = std::abs(buf[volx*j + i]);
     nrm2 += elem * elem;
    }
    nrm2 = std::sqrt(nrm2);
    for(DimOffset i = 0; i < volx; ++i){
     buf[volx*j + i] /= tensor_body_type(nrm2);
    }
    for(DimOffset k = j+1; k < voly; ++k){
     tensor_body_type dpr(0.0);
     for(DimOffset i = 0; i < volx; ++i){
      dpr += conjugated(buf[volx*j + i]) * buf[volx*k + i];
     }
     for(DimOffset i = 0; i < volx; ++i){
      buf[volx*k + i] -= dpr * buf[volx*j + i];
     }
    }
   }

   //Copy the result back into the tensor:
   rngy.reset();
   for(DimOffset j = 0; j < voly; ++j){
    rngx.reset();
    for(DimOffset i = 0; i < volx; ++i){
     tensor_body[rngy.globalOffset() + rngx.globalOffset()] = buf[volx*j + i];
     rngx.next();
    }
    rngy.next();
   }

   //Deallocate the temporary matricized buffer:
   delete [] buf;
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
