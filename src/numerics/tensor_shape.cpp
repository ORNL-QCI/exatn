/** ExaTN::Numerics: Tensor shape
REVISION: 2019/11/12

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_shape.hpp"

#include <iterator>
#include <cassert>

namespace exatn{

namespace numerics{

TensorShape::TensorShape()
{
}

TensorShape::TensorShape(const TensorShape & another,
                         const std::vector<unsigned int> & order):
 TensorShape(another)
{
 const auto rank = another.getRank();
 assert(order.size() == rank);
 const auto & orig = another.getDimExtents();
 for(unsigned int new_pos = 0; new_pos < rank; ++new_pos) extents_[new_pos] = orig[order[new_pos]];
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

void TensorShape::printItFile(std::ofstream & output_file) const
{
 output_file << "{";
 for(auto ext_it = extents_.cbegin(); ext_it != extents_.cend(); ++ext_it){
  if(std::next(ext_it,1) == extents_.cend()){
   output_file << *ext_it;
  }else{
   output_file << *ext_it << ",";
  }
 }
 output_file << "}";
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

bool TensorShape::isCongruentTo(const TensorShape & another) const
{
 const auto rank = this->getRank();
 if(another.getRank() != rank) return false;
 for(unsigned int i = 0; i < rank; ++i){
  if(this->getDimExtent(i) != another.getDimExtent(i)) return false;
 }
 return true;
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
