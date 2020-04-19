/** ExaTN::Numerics: Tensor range
REVISION: 2020/04/19

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
**/

#ifndef EXATN_NUMERICS_TENSOR_RANGE_HPP_
#define EXATN_NUMERICS_TENSOR_RANGE_HPP_

#include "tensor_basic.hpp"

#include <vector>

namespace exatn{

namespace numerics{

class TensorRange{
public:

 inline TensorRange(const std::vector<DimOffset> & bases,    //in: base offset of each dimension (0 is min)
                    const std::vector<DimExtent> & extents,  //in: extent of each dimension (on top of the base offset)
                    const std::vector<DimExtent> & strides); //in: stride of each dimension (global increment caused by local increment)

 TensorRange(const TensorRange &) = default;
 TensorRange & operator=(const TensorRange &) = default;
 TensorRange(TensorRange &&) noexcept = default;
 TensorRange & operator=(TensorRange &&) noexcept = default;
 ~TensorRange() = default;

 /** Resets the current multi-index value to the beginning. **/
 inline void reset();

 /** Returns the flat offset produced by the current multi-index value per se. **/
 inline DimOffset localOffset() const;

 /** Returns the flat offset produced by the current multi-index value within the global tensor range. **/
 inline DimOffset globalOffset() const;

 /** Increments the current multi-index value.
     If the tensor range is over, return false. **/
 inline bool next();

 /** Decrements the current multi-index value.
     If the tensor range is over, return false. **/
 inline bool prev();

private:

 std::vector<DimOffset> bases_;
 std::vector<DimExtent> extents_;
 std::vector<DimExtent> strides_;
 std::vector<DimOffset> mlndx_;
};


inline TensorRange::TensorRange(const std::vector<DimOffset> & bases,    //in: base offset of each dimension (0 is min)
                                const std::vector<DimExtent> & extents,  //in: extent of each dimension (on top of the base offset)
                                const std::vector<DimExtent> & strides): //in: stride of each dimension
 bases_(bases), extents_(extents), strides_(strides), mlndx_(bases.size())
{
 assert(bases.size() == extents.size() && bases.size() == strides.size());
}

inline void TensorRange::reset()
{
 for(auto & ind: mlndx_) ind = 0;
 return;
}

inline DimOffset TensorRange::localOffset() const
{
 DimOffset offset = 0;
 for(int i = extents_.size() - 1; i >= 0; --i) offset = offset * extents_[i] + mlndx_[i];
 return offset;
}

inline DimOffset TensorRange::globalOffset() const
{
 DimOffset offset = 0;
 for(int i = 0; i < extents_.size(); ++i) offset += (bases_[i] + mlndx_[i]) * strides_[i];
 return offset;
}

inline bool TensorRange::next()
{
 int rank = mlndx_.size();
 int i = 0;
 while(i < rank){
  if(mlndx_[i] + 1 < extents_[i]){
   mlndx_[i]++;
   return true;
  }else{
   mlndx_[i++] = 0;
  }
 }
 return false;
}

inline bool TensorRange::prev()
{
 int rank = mlndx_.size();
 int i = 0;
 while(i < rank){
  if(mlndx_[i] > 0){
   mlndx_[i]--;
   return true;
  }else{
   mlndx_[i++] = extents_[i] - 1;
  }
 }
 reset();
 return false;
}

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_RANGE_HPP_
