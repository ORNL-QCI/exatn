/** ExaTN::Numerics: Tensor factory
REVISION: 2019/03/13

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

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
  std::cout << "ERROR(TensorFactory:createTensor): Tensor kind TENSOR_SHA not yet implemented!" << std::endl;
 case TensorKind::TENSOR_EXA:
  std::cout << "ERROR(TensorFactory:createTensor): Tensor kind TENSOR_EXA not yet implemented!" << std::endl;
 default:
  std::cout << "ERROR(TensorFactory:createTensor): Unknown tensor kind!" << std::endl;
 };
 return std::unique_ptr<Tensor>(nullptr);
}

std::unique_ptr<Tensor> TensorFactory::createTensor(TensorKind tensor_kind,
                                                    const std::string & name,
                                                    const TensorShape & shape)
{
 switch(tensor_kind){
 case TensorKind::TENSOR:
  return std::unique_ptr<Tensor>(new Tensor(name,shape));
 case TensorKind::TENSOR_SHA:
  std::cout << "ERROR(TensorFactory:createTensor): Tensor kind TENSOR_SHA not yet implemented!" << std::endl;
 case TensorKind::TENSOR_EXA:
  std::cout << "ERROR(TensorFactory:createTensor): Tensor kind TENSOR_EXA not yet implemented!" << std::endl;
 default:
  std::cout << "ERROR(TensorFactory:createTensor): Unknown tensor kind!" << std::endl;
 };
 return std::unique_ptr<Tensor>(nullptr);
}

} //namespace numerics

} //namespace exatn
