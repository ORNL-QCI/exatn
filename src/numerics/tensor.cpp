/** ExaTN::Numerics: Tensor
REVISION: 2018/11/02

Copyright (C) 2018-2018 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2018 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor.hpp"

#include <assert.h>

#include <iostream>

namespace exatn{

namespace numerics{

Tensor::Tensor(const std::string & name,
               const TensorShape & shape,
               const TensorSignature & signature):
name_(name), shape_(shape), signature_(signature)
{
 if(signature_.getRank() != shape_.getRank()) std::cout << "ERROR(Tensor::Tensor): Signature/Shape size mismatch!" << std::endl;
 assert(signature_.getRank() == shape_.getRank());
}

Tensor::Tensor(const std::string & name,
               const TensorShape & shape):
name_(name), shape_(shape), signature_(shape.getRank())
{
}

void Tensor::printIt() const
{
 std::cout << name_; signature_.printIt(); shape_.printIt();
 return;
}

} //namespace numerics

} //namespace exatn
