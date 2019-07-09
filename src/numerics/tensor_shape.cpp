/** ExaTN::Numerics: Tensor shape
REVISION: 2019/07/08

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_shape.hpp"

#include <iostream>
#include <iterator>
#include <assert.h>

namespace exatn{

namespace numerics{

TensorShape::TensorShape()
{
}

void TensorShape::printIt() const
{
 std::cout << "{";
 for(auto ext_it = extents_.cbegin(); ext_it != extents_.cend(); ++ext_it){
  if(std::next(ext_it,1) == extents_.cend()){
   std::cout << *ext_it;
  }else{
   std::cout << *ext_it << ",";
  }
 }
 std::cout << "}";
 return;
}

unsigned int TensorShape::getRank() const
{
 return static_cast<unsigned int>(extents_.size());
}

DimExtent TensorShape::getDimExtent(unsigned int dim_id) const
{
 assert(dim_id < extents_.size()); //debug
 return extents_[dim_id];
}

const std::vector<DimExtent> & TensorShape::getDimExtents() const
{
 return extents_;
}

void TensorShape::resetDimension(unsigned int dim_id, DimExtent extent)
{
 assert(dim_id < extents_.size()); //debug
 extents_[dim_id] = extent;
 return;
}

void TensorShape::deleteDimension(unsigned int dim_id)
{
 assert(dim_id < extents_.size());
 extents_.erase(extents_.cbegin()+dim_id);
 return;
}

void TensorShape::appendDimension(DimExtent dim_extent)
{
 extents_.emplace_back(dim_extent);
 return;
}

} //namespace numerics

} //namespace exatn
