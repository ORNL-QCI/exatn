/** ExaTN::Numerics: Tensor Functor: Initialization to a random value
REVISION: 2022/04/12

Copyright (C) 2018-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle) **/

#include "functor_init_rnd.hpp"

#include "talshxx.hpp"

#include <functional>
#include <random>
#include <complex>

namespace exatn{

namespace numerics{

int FunctorInitRnd::apply(talsh::Tensor & local_tensor)
{
 std::random_device seeder;

 auto tensor_volume = local_tensor.getVolume();
 auto access_granted = false;

 {//Try REAL32:
  float * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted){
   if(random_seed_ == 0){
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-1.0,1.0);
    auto rnd = std::bind(distribution,generator);
#pragma omp parallel for schedule(guided) shared(tensor_volume,body)
    for(std::size_t i = 0; i < tensor_volume; ++i) body[i] = rnd();
   }else{
    std::default_random_engine generator(seeder());
    std::uniform_real_distribution<float> distribution(-1.0,1.0);
    auto rnd = std::bind(distribution,generator);
#pragma omp parallel for schedule(guided) shared(tensor_volume,body)
    for(std::size_t i = 0; i < tensor_volume; ++i) body[i] = rnd();
   }
   return 0;
  }
 }

 {//Try REAL64:
  double * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted){
   if(random_seed_ == 0){
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0,1.0);
    auto rnd = std::bind(distribution,generator);
#pragma omp parallel for schedule(guided) shared(tensor_volume,body)
    for(std::size_t i = 0; i < tensor_volume; ++i) body[i] = rnd();
   }else{
    std::default_random_engine generator(seeder());
    std::uniform_real_distribution<double> distribution(-1.0,1.0);
    auto rnd = std::bind(distribution,generator);
#pragma omp parallel for schedule(guided) shared(tensor_volume,body)
    for(std::size_t i = 0; i < tensor_volume; ++i) body[i] = rnd();
   }
   return 0;
  }
 }

 {//Try COMPLEX32:
  std::complex<float> * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted){
   if(random_seed_ == 0){
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-1.0,1.0);
    auto rnd = std::bind(distribution,generator);
    if(real_only_){
#pragma omp parallel for schedule(guided) shared(tensor_volume,body)
     for(std::size_t i = 0; i < tensor_volume; ++i) body[i] = std::complex<float>{rnd(),0.0f};
    }else{
#pragma omp parallel for schedule(guided) shared(tensor_volume,body)
     for(std::size_t i = 0; i < tensor_volume; ++i) body[i] = std::complex<float>{rnd(),rnd()};
    }
   }else{
    std::default_random_engine generator(seeder());
    std::uniform_real_distribution<float> distribution(-1.0,1.0);
    auto rnd = std::bind(distribution,generator);
    if(real_only_){
#pragma omp parallel for schedule(guided) shared(tensor_volume,body)
     for(std::size_t i = 0; i < tensor_volume; ++i) body[i] = std::complex<float>{rnd(),0.0f};
    }else{
#pragma omp parallel for schedule(guided) shared(tensor_volume,body)
     for(std::size_t i = 0; i < tensor_volume; ++i) body[i] = std::complex<float>{rnd(),rnd()};
    }
   }
   return 0;
  }
 }

 {//Try COMPLEX64:
  std::complex<double> * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted){
   if(random_seed_ == 0){
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0,1.0);
    auto rnd = std::bind(distribution,generator);
    if(real_only_){
#pragma omp parallel for schedule(guided) shared(tensor_volume,body)
     for(std::size_t i = 0; i < tensor_volume; ++i) body[i] = std::complex<double>{rnd(),0.0};
    }else{
#pragma omp parallel for schedule(guided) shared(tensor_volume,body)
     for(std::size_t i = 0; i < tensor_volume; ++i) body[i] = std::complex<double>{rnd(),rnd()};
    }
   }else{
    std::default_random_engine generator(seeder());
    std::uniform_real_distribution<double> distribution(-1.0,1.0);
    auto rnd = std::bind(distribution,generator);
    if(real_only_){
#pragma omp parallel for schedule(guided) shared(tensor_volume,body)
     for(std::size_t i = 0; i < tensor_volume; ++i) body[i] = std::complex<double>{rnd(),0.0};
    }else{
#pragma omp parallel for schedule(guided) shared(tensor_volume,body)
     for(std::size_t i = 0; i < tensor_volume; ++i) body[i] = std::complex<double>{rnd(),rnd()};
    }
   }
   return 0;
  }
 }

 std::cout << "#ERROR(exatn::numerics::FunctorInitRnd): Unknown data kind in talsh::Tensor!" << std::endl;
 return 1;
}

} //namespace numerics

} //namespace exatn
