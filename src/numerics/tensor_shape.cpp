/** ExaTN::Numerics: Tensor shape
REVISION: 2018/12/18

Copyright (C) 2018-2018 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2018 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_shape.hpp"

#include <iostream>
#include <iterator>

namespace exatn{

namespace numerics{

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

} //namespace numerics

} //namespace exatn
