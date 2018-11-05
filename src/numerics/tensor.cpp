/** ExaTN::Numerics: Tensor
REVISION: 2018/11/04

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

unsigned int Tensor::getRank() const
{
 return this->shape_.getRank();
}

DimExtent Tensor::getDimExtent(unsigned int dim_id) const
{
 return this->shape_.getDimExtent(dim_id);
}

SpaceId Tensor::getDimSpaceId(unsigned int dim_id) const
{
 return this->signature_.getDimSpaceId(dim_id);
}

SubspaceId Tensor::getDimSubspaceId(unsigned int dim_id) const
{
 return this->signature_.getDimSubspaceId(dim_id);
}

std::pair<SpaceId,SubspaceId> Tensor::getDimSpaceAttr(unsigned int dim_id) const
{
 return this->signature_.getDimSpaceAttr(dim_id);
}

} //namespace numerics

} //namespace exatn
