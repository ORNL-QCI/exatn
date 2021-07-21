/** ExaTN::Numerics: Tensor Functor: Computes 1-norm of a tensor
REVISION: 2021/07/21

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "functor_norm1.hpp"

#include "talshxx.hpp"

namespace exatn{

namespace numerics{

std::mutex FunctorNorm1::mutex_;

int FunctorNorm1::apply(talsh::Tensor & local_tensor)
{
 const std::lock_guard<std::mutex> lock(mutex_);
 //norm_ = 0.0;
 const auto tensor_volume = local_tensor.getVolume();
 auto access_granted = false;

 auto norm1_func = [&](const auto * tensor_body){
  double norm = 0.0;
#pragma omp parallel for schedule(guided) shared(tensor_volume,tensor_body) reduction(+:norm)
  for(std::size_t i = 0; i < tensor_volume; ++i)
   norm += static_cast<double>(std::abs(tensor_body[i]));
  return norm;
 };

 {//Try REAL32:
  const float * body;
  access_granted = local_tensor.getDataAccessHostConst(&body);
  if(access_granted){
   norm_ += norm1_func(body);
   return 0;
  }
 }

 {//Try REAL64:
  const double * body;
  access_granted = local_tensor.getDataAccessHostConst(&body);
  if(access_granted){
   norm_ += norm1_func(body);
   return 0;
  }
 }

 {//Try COMPLEX32:
  const std::complex<float> * body;
  access_granted = local_tensor.getDataAccessHostConst(&body);
  if(access_granted){
   norm_ += norm1_func(body);
   return 0;
  }
 }

 {//Try COMPLEX64:
  const std::complex<double> * body;
  access_granted = local_tensor.getDataAccessHostConst(&body);
  if(access_granted){
   norm_ += norm1_func(body);
   return 0;
  }
 }

 std::cout << "#ERROR(exatn::numerics::FunctorNorm1): Unknown data kind in talsh::Tensor!" << std::endl;
 return 1;
}

} //namespace numerics

} //namespace exatn
