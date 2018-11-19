/** ExaTN::Numerics: Tensor factory
REVISION: 2018/11/18

Copyright (C) 2018-2018 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2018 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_factory.hpp"

namespace exatn{

namespace numerics{

std::unique_ptr<Tensor> TensorFactory::createTensor(TensorKind tensor_kind,
                                                    const std::string & name,
                                                    const TensorShape & shape,
                                                    const TensorSignature & signature)
{
 switch(tensor_kind){
 case TensorKind::TENSOR:
  return std::unique_ptr<Tensor>(new Tensor(name,shape,signature));
 case TensorKind::TENSOR_SHA:
  std::cout << "ERROR(TensorFactory:createTensor): Tensor kind TENSOR_SHA not yet implemented!" << std::endl; assert(false);
 case TensorKind::TENSOR_EXA:
  std::cout << "ERROR(TensorFactory:createTensor): Tensor kind TENSOR_EXA not yet implemented!" << std::endl; assert(false);
 default:
  std::cout << "ERROR(TensorFactory:createTensor): Unknown tensor kind!" << std::endl; assert(false);
 };
}

std::unique_ptr<Tensor> TensorFactory::createTensor(TensorKind tensor_kind,
                                                    const std::string & name,
                                                    const TensorShape & shape)
{
 switch(tensor_kind){
 case TensorKind::TENSOR:
  return std::unique_ptr<Tensor>(new Tensor(name,shape));
 case TensorKind::TENSOR_SHA:
  std::cout << "ERROR(TensorFactory:createTensor): Tensor kind TENSOR_SHA not yet implemented!" << std::endl; assert(false);
 case TensorKind::TENSOR_EXA:
  std::cout << "ERROR(TensorFactory:createTensor): Tensor kind TENSOR_EXA not yet implemented!" << std::endl; assert(false);
 default:
  std::cout << "ERROR(TensorFactory:createTensor): Unknown tensor kind!" << std::endl; assert(false);
 };
}

} //namespace numerics

} //namespace exatn
