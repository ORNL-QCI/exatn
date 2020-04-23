/** ExaTN::Numerics: Tensor range
REVISION: 2020/04/23

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Tensor range is a Cartesian product of one or more ranges.
 (b) Each constituent range can be a subrange of a larger range,
     thus requiring a base offset and extent for its specification.
 (c) Each tensor range can be flattened into a single 1d super-range
     by a standard column-wise (little endian) mapping such that
     a unique local offset in this 1d super-range can be associated
     with each multi-index from within the tensor range.
 (d) If a tensor range contains at least one subrange in its definition
     the striges must be provided during tensor range construction to
     indicate this: A stride of index J is equal to the increment in
     the 1d super-range associated with the parental tensor range when
     index J is incremented by one in the current tensor range.
     Thus, each multi-index from within the current tensor range
     is not only associated with its local offset but also with its
     global offset in the parental flattened 1d super-range, which
     is equal to the local offset when the tensor range does not
     contain subranges (tensor range = parental tensor range).
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

 inline TensorRange(const std::vector<DimOffset> & bases,    //in: base offset of each dimension (0 is min)
                    const std::vector<DimExtent> & extents); //in: extent of each dimension (on top of the base offset)

 inline TensorRange(const std::vector<DimExtent> & extents); //in: extent of each dimension (base offset = 0)

 TensorRange(const TensorRange &) = default;
 TensorRange & operator=(const TensorRange &) = default;
 TensorRange(TensorRange &&) noexcept = default;
 TensorRange & operator=(TensorRange &&) noexcept = default;
 ~TensorRange() = default;

 /** Returns the tensor range volume (local volume). **/
 inline DimExtent localVolume() const;

 /** Returns the parental tensor range volume (global volume). **/
 inline DimExtent globalVolume() const;

 /** Resets the current multi-index value to the beginning. **/
 inline void reset();

 /** Retrieves a specific index from the multi-index. **/
 inline DimOffset getIndex(unsigned int position) const;

 /** Returns the flat offset produced by the current multi-index value per se. **/
 inline DimOffset localOffset() const; //little endian

 /** Returns the flat offset produced by the current multi-index value within the global tensor range. **/
 inline DimOffset globalOffset() const; //based on strides

 /** Increments the current multi-index value.
     If the tensor range is over, return false. **/
 inline bool next(DimOffset increment = 1);

 /** Decrements the current multi-index value.
     If the tensor range is over, return false. **/
 inline bool prev(DimOffset increment = 1);

private:

 std::vector<DimOffset> bases_;
 std::vector<DimExtent> extents_;
 std::vector<DimExtent> strides_;
 std::vector<DimOffset> mlndx_;
 DimExtent volume_;
};


inline TensorRange::TensorRange(const std::vector<DimOffset> & bases,    //in: base offset of each dimension (0 is min)
                                const std::vector<DimExtent> & extents,  //in: extent of each dimension (on top of the base offset)
                                const std::vector<DimExtent> & strides): //in: stride of each dimension
 bases_(bases), extents_(extents), strides_(strides), mlndx_(bases.size())
{
 assert(bases_.size() == extents_.size() && bases_.size() == strides_.size());
 if(extents_.size() > 0){
  volume_ = 1; for(const auto & extent: extents_) volume_ *= extent;
 }else{
  volume_ = 0;
 }
 reset();
}


inline TensorRange::TensorRange(const std::vector<DimOffset> & bases,    //in: base offset of each dimension (0 is min)
                                const std::vector<DimExtent> & extents): //in: extent of each dimension (on top of the base offset)
 bases_(bases), extents_(extents), strides_(bases.size()), mlndx_(bases.size())
{
 assert(bases_.size() == extents_.size());
 if(extents_.size() > 0){
  volume_ = 1;
  for(unsigned int i = 0; i < extents_.size(); ++i){
   strides_[i] = volume_;
   volume_ *= extents_[i];
  }
 }else{
  volume_ = 0;
 }
 reset();
}


inline TensorRange::TensorRange(const std::vector<DimExtent> & extents): //in: extent of each dimension (base offset = 0)
 bases_(extents.size(),0), extents_(extents), strides_(extents.size()), mlndx_(extents.size())
{
 if(extents_.size() > 0){
  volume_ = 1;
  for(unsigned int i = 0; i < extents_.size(); ++i){
   strides_[i] = volume_;
   volume_ *= extents_[i];
  }
 }else{
  volume_ = 0;
 }
 reset();
}


inline DimExtent TensorRange::localVolume() const
{
 return volume_;
}


inline DimExtent TensorRange::globalVolume() const
{
 const auto range_rank = extents_.size();
 if(range_rank == 0) return 0;
 return strides_[range_rank-1] * extents_[range_rank-1];
}


inline void TensorRange::reset()
{
 for(auto & ind: mlndx_) ind = 0;
 return;
}


inline DimOffset TensorRange::getIndex(unsigned int position) const
{
 assert(position < mlndx_.size());
 return mlndx_[position];
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


inline bool TensorRange::next(DimOffset increment)
{
 const int rank = mlndx_.size();
 int i = 0;
 while(i < rank){
  if(mlndx_[i] + 1 < extents_[i]){
   mlndx_[i]++;
   if(--increment == 0) return true;
   i = 0;
  }else{
   mlndx_[i++] = 0;
  }
 }
 return false;
}


inline bool TensorRange::prev(DimOffset increment)
{
 const int rank = mlndx_.size();
 int i = 0;
 while(i < rank){
  if(mlndx_[i] > 0){
   mlndx_[i]--;
   if(--increment == 0) return true;
   i = 0;
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
