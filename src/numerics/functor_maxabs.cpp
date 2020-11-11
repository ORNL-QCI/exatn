/** ExaTN::Numerics: Tensor Functor: Computes max-abs norm of a tensor
REVISION: 2020/11/11

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "functor_maxabs.hpp"

#include "talshxx.hpp"

namespace exatn{

namespace numerics{

int FunctorMaxAbs::apply(talsh::Tensor & local_tensor)
{
 norm_ = 0.0;
 const auto tensor_volume = local_tensor.getVolume();
 auto access_granted = false;

 auto maxabs_func = [&](const auto * tensor_body){
  double norm = 0.0;
#pragma omp parallel for schedule(guided) shared(tensor_volume,tensor_body) reduction(max:norm)
  for(std::size_t i = 0; i < tensor_volume; ++i){
   const auto absval = static_cast<double>(std::abs(tensor_body[i]));
   if(absval > norm) norm = absval;
  }
  return norm;
 };

 {//Try REAL32:
  const float * body;
  access_granted = local_tensor.getDataAccessHostConst(&body);
  if(access_granted){
   norm_ = maxabs_func(body);
   return 0;
  }
 }

 {//Try REAL64:
  const double * body;
  access_granted = local_tensor.getDataAccessHostConst(&body);
  if(access_granted){
   norm_ = maxabs_func(body);
   return 0;
  }
 }

 {//Try COMPLEX32:
  const std::complex<float> * body;
  access_granted = local_tensor.getDataAccessHostConst(&body);
  if(access_granted){
   norm_ = maxabs_func(body);
   return 0;
  }
 }

 {//Try COMPLEX64:
  const std::complex<double> * body;
  access_granted = local_tensor.getDataAccessHostConst(&body);
  if(access_granted){
   norm_ = maxabs_func(body);
   return 0;
  }
 }

 std::cout << "#ERROR(exatn::numerics::FunctorMaxAbs): Unknown data kind in talsh::Tensor!" << std::endl;
 return 1;
}

} //namespace numerics

} //namespace exatn
