/** ExaTN::Numerics: Tensor Functor: Checks the tensor on the presence of NaN
REVISION: 2022/09/02

Copyright (C) 2018-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle)
Copyright (C) 2022-2022 NVIDIA Corporation

SPDX-License-Identifier: BSD-3-Clause **/

#include "functor_isnan.hpp"

#include "talshxx.hpp"

namespace exatn{

namespace numerics{

std::mutex FunctorIsNaN::mutex_;

int FunctorIsNaN::apply(talsh::Tensor & local_tensor)
{
 const std::lock_guard<std::mutex> lock(mutex_);

 const auto tensor_volume = local_tensor.getVolume();
 auto access_granted = false;

 auto num_nans_func = [&](const auto * tensor_body){
  std::size_t num_nans = 0;
#pragma omp parallel for schedule(guided) shared(tensor_volume,tensor_body) reduction(+:num_nans)
  for(std::size_t i = 0; i < tensor_volume; ++i){
   if(isnan(tensor_body[i])) ++num_nans;
  }
  return num_nans;
 };

 {//Try REAL32:
  const float * body;
  access_granted = local_tensor.getDataAccessHostConst(&body);
  if(access_granted){
   num_nans_ = num_nans_func(body);
   return 0;
  }
 }

 {//Try REAL64:
  const double * body;
  access_granted = local_tensor.getDataAccessHostConst(&body);
  if(access_granted){
   num_nans_ = num_nans_func(body);
   return 0;
  }
 }

 {//Try COMPLEX32:
  const std::complex<float> * body;
  access_granted = local_tensor.getDataAccessHostConst(&body);
  if(access_granted){
   num_nans_ = num_nans_func(body);
   return 0;
  }
 }

 {//Try COMPLEX64:
  const std::complex<double> * body;
  access_granted = local_tensor.getDataAccessHostConst(&body);
  if(access_granted){
   num_nans_ = num_nans_func(body);
   return 0;
  }
 }

 std::cout << "#ERROR(exatn::numerics::FunctorIsNaN): Unknown data kind in talsh::Tensor!" << std::endl;
 return 1;
}

} //namespace numerics

} //namespace exatn
