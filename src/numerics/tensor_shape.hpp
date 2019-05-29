/** ExaTN::Numerics: Tensor shape
REVISION: 2019/05/27

Copyright (C) 2018-2018 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2018 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Tensor shape is an ordered set of tensor dimension extents.
     A scalar tensor (rank-0 tensor) has an empty shape.
**/

#ifndef EXATN_NUMERICS_TENSOR_SHAPE_HPP_
#define EXATN_NUMERICS_TENSOR_SHAPE_HPP_

#include "tensor_basic.hpp"

#include <assert.h>

#include <iostream>
#include <type_traits>
#include <initializer_list>
#include <vector>

namespace exatn{

namespace numerics{

class TensorShape{
public:

 /** Create a tensor shape by specifying extents for all tensor dimensions. **/
 template<typename T>
 TensorShape(std::initializer_list<T> extents);
 template<typename T>
 TensorShape(const std::vector<T> & extents);

 TensorShape(const TensorShape & tens_shape) = default;
 TensorShape & operator=(const TensorShape & tens_shape) = default;
 TensorShape(TensorShape && tens_shape) noexcept = default;
 TensorShape & operator=(TensorShape && tens_shape) noexcept = default;
 virtual ~TensorShape() = default;

 /** Print. **/
 void printIt() const;

 /** Get tensor rank (number of tensor dimensions). **/
 unsigned int getRank() const;

 /** Get the extent of a specific tensor dimension. **/
 DimExtent getDimExtent(unsigned int dim_id) const;

 /** Get the extents of all tensor dimensions. **/
 const std::vector<DimExtent> & getDimExtents() const;

private:

 std::vector<DimExtent> extents_; //tensor dimension extents
};


//TEMPLATES:
template<typename T>
TensorShape::TensorShape(std::initializer_list<T> extents):
extents_(extents.size())
{
 static_assert(std::is_integral<T>::value,"FATAL(TensorShape::TensorShape): TensorShape extent type must be integral!");

 //DEBUG:
 for(const auto & extent: extents){
  if(extent < 0) std::cout << "ERROR(TensorShape::TensorShape): Negative dimension extent passed!" << std::endl;
  assert(extent >= 0);
 }

 int i = 0;
 for(const auto & extent: extents) extents_[i++] = DimExtent{extent};
}

template<typename T>
TensorShape::TensorShape(const std::vector<T> & extents):
extents_(extents.size())
{
 static_assert(std::is_integral<T>::value,"FATAL(TensorShape::TensorShape): TensorShape extent type must be integral!");

 //DEBUG:
 for(const auto & extent: extents){
  if(extent < 0) std::cout << "ERROR(TensorShape::TensorShape): Negative dimension extent passed!" << std::endl;
  assert(extent >= 0);
 }

 int i = 0;
 for(const auto & extent: extents) extents_[i++] = DimExtent{extent};
}

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_SHAPE_HPP_
