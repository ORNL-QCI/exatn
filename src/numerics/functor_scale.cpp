/** ExaTN::Numerics: Tensor Functor: Scaling a tensor by a scalar
REVISION: 2019/11/21

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "functor_scale.hpp"

#include "talshxx.hpp"

namespace exatn{

namespace numerics{

int FunctorScale::apply(talsh::Tensor & local_tensor)
{
 auto tensor_volume = local_tensor.getVolume();
 auto access_granted = false;

 {//Try REAL32:
  float * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted){
   const auto val = static_cast<float>(scale_val_.real());
#pragma omp parallel for schedule(guided) shared(tensor_volume,body,val)
   for(std::size_t i = 0; i < tensor_volume; ++i) body[i] *= val;
   return 0;
  }
 }

 {//Try REAL64:
  double * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted){
   const auto val = static_cast<double>(scale_val_.real());
#pragma omp parallel for schedule(guided) shared(tensor_volume,body,val)
   for(std::size_t i = 0; i < tensor_volume; ++i) body[i] *= val;
   return 0;
  }
 }

 {//Try COMPLEX32:
  std::complex<float> * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted){
   const auto val = std::complex<float>{static_cast<float>(scale_val_.real()),
                                        static_cast<float>(scale_val_.imag())};
#pragma omp parallel for schedule(guided) shared(tensor_volume,body,val)
   for(std::size_t i = 0; i < tensor_volume; ++i) body[i] *= val;
   return 0;
  }
 }

 {//Try COMPLEX64:
  std::complex<double> * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted){
   const auto val = scale_val_;
#pragma omp parallel for schedule(guided) shared(tensor_volume,body,val)
   for(std::size_t i = 0; i < tensor_volume; ++i) body[i] *= val;
   return 0;
  }
 }

 std::cout << "#ERROR(exatn::numerics::FunctorScale): Unknown data kind in talsh::Tensor!" << std::endl;
 return 1;
}

} //namespace numerics

} //namespace exatn
