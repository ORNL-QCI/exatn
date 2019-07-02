/** ExaTN::Numerics: Tensor
REVISION: 2019/07/02

Copyright (C) 2018-2018 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2018 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor.hpp"

#include <iostream>
#include <assert.h>

namespace exatn{

namespace numerics{

Tensor::Tensor(const std::string & name,
               const TensorShape & shape,
               const TensorSignature & signature):
name_(name), shape_(shape), signature_(signature)
{
 //DEBUG:
 if(signature_.getRank() != shape_.getRank()) std::cout << "ERROR(Tensor::Tensor): Signature/Shape size mismatch!" << std::endl;
 assert(signature_.getRank() == shape_.getRank());
}

Tensor::Tensor(const std::string & name,
               const TensorShape & shape):
name_(name), shape_(shape), signature_(shape.getRank())
{
}

Tensor::Tensor(const std::string & name):
name_(name)
{
}

void Tensor::printIt() const
{
 std::cout << name_; signature_.printIt(); shape_.printIt();
 return;
}

unsigned int Tensor::getRank() const
{
 return shape_.getRank();
}

const TensorShape & Tensor::getShape() const
{
 return shape_;
}

const TensorSignature & Tensor::getSignature() const
{
 return signature_;
}

DimExtent Tensor::getDimExtent(unsigned int dim_id) const
{
 return shape_.getDimExtent(dim_id);
}

const std::vector<DimExtent> & Tensor::getDimExtents() const
{
 return shape_.getDimExtents();
}

SpaceId Tensor::getDimSpaceId(unsigned int dim_id) const
{
 return signature_.getDimSpaceId(dim_id);
}

SubspaceId Tensor::getDimSubspaceId(unsigned int dim_id) const
{
 return signature_.getDimSubspaceId(dim_id);
}

std::pair<SpaceId,SubspaceId> Tensor::getDimSpaceAttr(unsigned int dim_id) const
{
 return signature_.getDimSpaceAttr(dim_id);
}

void Tensor::deleteDimension(unsigned int dim_id)
{
 signature_.deleteDimension(dim_id);
 shape_.deleteDimension(dim_id);
 return;
}

void Tensor::appendDimension(std::pair<SpaceId,SubspaceId> subspace, DimExtent dim_extent)
{
 signature_.appendDimension(subspace);
 shape_.appendDimension(dim_extent);
 return;
}

void Tensor::appendDimension(DimExtent dim_extent)
{
 this->appendDimension(std::pair<SpaceId,SubspaceId>{SOME_SPACE,0},dim_extent);
 return;
}

std::size_t Tensor::getTensorId() const
{
 return reinterpret_cast<std::size_t>(this);
}

} //namespace numerics

} //namespace exatn
