/** ExaTN::Numerics: Tensor range
REVISION: 2022/06/13

Copyright (C) 2018-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle)
Copyright (C) 2022-2022 NVIDIA Corporation

SPDX-License-Identifier: BSD-3-Clause **/

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
     Note that only generalized column-wise strides are allowed.
 (e) A tensor range can also be split into disjoint chunks such
     that each chunk can be iterated over by a concurrent agent.
**/

#ifndef EXATN_NUMERICS_TENSOR_RANGE_HPP_
#define EXATN_NUMERICS_TENSOR_RANGE_HPP_

#include "tensor_basic.hpp"

#include <vector>
#include <iostream>
#include <fstream>

#include "errors.hpp"

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

 template <typename IntegralType>
 inline TensorRange(const unsigned int num_dims,   //in: number of dimensions
                    const IntegralType extents[]); //in: dimension extents (base offsets = 0)

 TensorRange(const TensorRange &) = default;
 TensorRange & operator=(const TensorRange &) = default;
 TensorRange(TensorRange &&) noexcept = default;
 TensorRange & operator=(TensorRange &&) noexcept = default;
 ~TensorRange() = default;

 /** Returns the tensor range volume (local volume). **/
 inline DimExtent localVolume() const;

 /** Returns the parental tensor range volume (global volume). **/
 //inline DimExtent globalVolume() const;

 /** Resets the current multi-index value to the beginning. **/
 inline void reset();

 /** Resets the current multi-index value to a given flattened local offset. **/
 inline void reset(DimOffset offset);

 /** Resets the current multi-index value to a given (local) multi-index value. **/
 inline void reset(const std::vector<DimOffset> & mlndx);

 /** Resets the current multi-index for a number of concurrent progress agents,
     each given an exclusive subrange of this range to iterate within. Returns
     TRUE on success, FALSE if the subrange is empty for the current progress agent. **/
 inline bool reset(unsigned int num_agents,  //number of concurrent agents (iterators)
                   unsigned int agent_rank); //current agend id: [0..num_agents-1]

 /** Returns the current multi-index value. **/
 inline const std::vector<DimOffset> & getMultiIndex() const;

 /** Retrieves a specific index from the multi-index. **/
 inline DimOffset getIndex(unsigned int position) const;
 inline DimOffset operator[](unsigned int position) const;

 /** Tests whether all indices in the current multi-index have the same value. **/
 inline bool onDiagonal() const;

 /** Tests whether the indices of the current multi-index are in a monotonically increasing order. **/
 inline bool increasingOrder() const;

 /** Tests whether the indices of the current multi-index are in a monotonically decreasing order. **/
 inline bool decreasingOrder() const;

 /** Tests whether the indices of the current multi-index are in a non-decreasing order. **/
 inline bool nondecreasingOrder() const;

 /** Tests whether the indices of the current multi-index are in a non-increasing order. **/
 inline bool nonincreasingOrder() const;

 /** Tests whether both halves of the indices of the current multi-index are the same and
     each is in a monotonically increasing order. **/
 inline bool increasingOrderDiag() const;

 /** Tests whether both halves of the indices of the current multi-index are the same and
     each is in a monotonically decreasing order. **/
 inline bool decreasingOrderDiag() const;

 /** Tests whether both halves of the indices of the current multi-index are the same and
     each is in a monotonically non-decreasing order. **/
 inline bool nondecreasingOrderDiag() const;

 /** Tests whether both halves of the indices of the current multi-index are the same and
     each is in a monotonically non-increasing order. **/
 inline bool nonincreasingOrderDiag() const;

 /** Returns the flattened offset produced by the current multi-index value per se. **/
 inline DimOffset localOffset() const; //little endian

 /** Returns the flattened offset produced by the current multi-index value within the global tensor range. **/
 inline DimOffset globalOffset() const; //based on strides

 /** Increments the current multi-index value.
     If the tensor range (or subrange) is over, return false. **/
 inline bool next(DimOffset increment = 1); //increment value

 /** Decrements the current multi-index value.
     If the tensor range (or subrange) is over, return false. **/
 inline bool prev(DimOffset increment = 1); //increment value

 /** Shifts the current multi-index value by a given flattend local offset.
     If the tensor range (or subrange) is over, return false. **/
 inline bool shift(long long offset);

 /** Prints the current multi-index value. **/
 void printCurrent() const{
  std::cout << "{";
  for(const auto & ind: mlndx_) std::cout << " " << ind;
  std::cout << " }" << std::flush;
  return;
 }

 /** Prints the current multi-index value to file. **/
 void printCurrent(std::ofstream & output_file) const{
  output_file << "{";
  for(const auto & ind: mlndx_) output_file << " " << ind;
  output_file << " }";
  //output_file.flush();
  return;
 }

private:

 std::vector<DimOffset> bases_;   //local base offsets
 std::vector<DimExtent> extents_; //local extents on top of base offsets
 std::vector<DimExtent> strides_; //strides (based on global extents)
 std::vector<DimOffset> mlndx_;   //current local multi-index value
 DimExtent volume_;               //total local volume
 DimOffset subrange_begin_;       //subrange begin (if multiple agents iterate through the range)
 DimOffset subrange_end_;         //subrange end (if multiple agents iterate through the range)
};


inline TensorRange::TensorRange(const std::vector<DimOffset> & bases,    //in: base offset of each dimension (0 is min)
                                const std::vector<DimExtent> & extents,  //in: extent of each dimension (on top of the base offset)
                                const std::vector<DimExtent> & strides): //in: stride of each dimension
 bases_(bases), extents_(extents), strides_(strides), mlndx_(extents.size()), volume_(0)
{
 assert(bases_.size() == extents_.size() && strides_.size() == extents_.size());
 if(extents_.size() > 0){
  for(unsigned int i = 1; i < extents_.size(); ++i) assert(strides_[i] >= strides_[i-1]);
  volume_ = 1; for(const auto & extent: extents_) volume_ *= extent;
 }
 reset();
}


inline TensorRange::TensorRange(const std::vector<DimOffset> & bases,    //in: base offset of each dimension (0 is min)
                                const std::vector<DimExtent> & extents): //in: extent of each dimension (on top of the base offset)
 bases_(bases), extents_(extents), strides_(extents.size()), mlndx_(extents.size()), volume_(0)
{
 assert(bases_.size() == extents_.size());
 if(extents_.size() > 0){
  DimExtent global_volume = 1;
  volume_ = 1;
  for(unsigned int i = 0; i < extents_.size(); ++i){
   strides_[i] = global_volume;
   global_volume *= (bases_[i] + extents_[i]);
   volume_ *= extents_[i];
  }
 }
 reset();
}


inline TensorRange::TensorRange(const std::vector<DimExtent> & extents): //in: extent of each dimension (base offset = 0)
 bases_(extents.size(),0), extents_(extents), strides_(extents.size()), mlndx_(extents.size()), volume_(0)
{
 if(extents_.size() > 0){
  volume_ = 1;
  for(unsigned int i = 0; i < extents_.size(); ++i){
   strides_[i] = volume_;
   volume_ *= extents_[i];
  }
 }
 reset();
}


template <typename IntegralType>
inline TensorRange::TensorRange(const unsigned int num_dims,   //in: number of dimensions
                                const IntegralType extents[]): //in: dimension extents (base offsets = 0)
 bases_(num_dims,0), extents_(num_dims,0), strides_(num_dims), mlndx_(num_dims), volume_(0)
{
 if(num_dims > 0){
  for(unsigned int i = 0; i < num_dims; ++i) extents_[i] = extents[i];
  volume_ = 1;
  for(unsigned int i = 0; i < num_dims; ++i){
   strides_[i] = volume_;
   volume_ *= extents_[i];
  }
 }
 reset();
}


inline DimExtent TensorRange::localVolume() const
{
 return volume_;
}


inline void TensorRange::reset()
{
 for(auto & ind: mlndx_) ind = 0;
 subrange_begin_ = volume_;
 subrange_end_ = 0;
 return;
}


inline void TensorRange::reset(DimOffset offset)
{
 assert(offset < volume_);
 reset();
 shift(static_cast<long long>(offset));
 return;
}


inline void TensorRange::reset(const std::vector<DimOffset> & mlndx)
{
 const unsigned int rank = mlndx.size();
 assert(rank == mlndx_.size());
 reset();
 for(unsigned int i = 0; i < rank; ++i){
  mlndx_[i] = mlndx[i];
  assert(mlndx_[i] < extents_[i]);
 }
 return;
}


inline bool TensorRange::reset(unsigned int num_agents,
                               unsigned int agent_rank)
{
 if(volume_ == 0) return false;
 auto chunk = volume_ / num_agents;
 auto remainder = volume_ % num_agents;
 subrange_begin_ = chunk * agent_rank + std::min(static_cast<decltype(remainder)>(agent_rank),remainder);
 if(agent_rank < remainder) chunk++;
 subrange_end_ = subrange_begin_ + chunk;
 if(chunk == 0) return false;
 auto offs = subrange_begin_;
 for(unsigned int i = 0; i < extents_.size(); ++i){
  mlndx_[i] = offs % extents_[i];
  offs /= extents_[i];
 }
 return true;
}


inline const std::vector<DimOffset> & TensorRange::getMultiIndex() const
{
 return mlndx_;
}


inline DimOffset TensorRange::getIndex(unsigned int position) const
{
 assert(position < mlndx_.size());
 return mlndx_[position];
}


inline DimOffset TensorRange::operator[](unsigned int position) const
{
 assert(position < mlndx_.size());
 return mlndx_[position];
}


inline bool TensorRange::onDiagonal() const
{
 bool diag = true;
 for(int i = 1; i < mlndx_.size(); ++i){
  if((bases_[i] + mlndx_[i]) != (bases_[i-1] + mlndx_[i-1])){
   diag = false;
   break;
  }
 }
 return diag;
}


inline bool TensorRange::increasingOrder() const
{
 bool indeed = true;
 for(int i = 1; i < mlndx_.size(); ++i){
  if((bases_[i] + mlndx_[i]) <= (bases_[i-1] + mlndx_[i-1])){
   indeed = false;
   break;
  }
 }
 return indeed;
}


inline bool TensorRange::decreasingOrder() const
{
 bool indeed = true;
 for(int i = 1; i < mlndx_.size(); ++i){
  if((bases_[i] + mlndx_[i]) >= (bases_[i-1] + mlndx_[i-1])){
   indeed = false;
   break;
  }
 }
 return indeed;
}


inline bool TensorRange::nondecreasingOrder() const
{
 bool indeed = true;
 for(int i = 1; i < mlndx_.size(); ++i){
  if((bases_[i] + mlndx_[i]) < (bases_[i-1] + mlndx_[i-1])){
   indeed = false;
   break;
  }
 }
 return indeed;
}


inline bool TensorRange::nonincreasingOrder() const
{
 bool indeed = true;
 for(int i = 1; i < mlndx_.size(); ++i){
  if((bases_[i] + mlndx_[i]) > (bases_[i-1] + mlndx_[i-1])){
   indeed = false;
   break;
  }
 }
 return indeed;
}


inline bool TensorRange::increasingOrderDiag() const
{
 bool indeed = (mlndx_.size() % 2 == 0);
 int half_size = mlndx_.size() / 2;
 if(indeed && half_size > 0){
  for(int i = 1; i < half_size; ++i){
   if(((bases_[i] + mlndx_[i]) <= (bases_[i-1] + mlndx_[i-1])) ||
      ((bases_[i] + mlndx_[i]) != (bases_[half_size+i] + mlndx_[half_size+i]))){
    indeed = false;
    break;
   }
  }
  indeed = indeed && ((bases_[0] + mlndx_[0]) == (bases_[half_size] + mlndx_[half_size]));
 }
 return indeed;
}


inline bool TensorRange::decreasingOrderDiag() const
{
 bool indeed = (mlndx_.size() % 2 == 0);
 int half_size = mlndx_.size() / 2;
 if(indeed && half_size > 0){
  for(int i = 1; i < half_size; ++i){
   if(((bases_[i] + mlndx_[i]) >= (bases_[i-1] + mlndx_[i-1])) ||
      ((bases_[i] + mlndx_[i]) != (bases_[half_size+i] + mlndx_[half_size+i]))){
    indeed = false;
    break;
   }
  }
  indeed = indeed && ((bases_[0] + mlndx_[0]) == (bases_[half_size] + mlndx_[half_size]));
 }
 return indeed;
}


inline bool TensorRange::nondecreasingOrderDiag() const
{
 bool indeed = (mlndx_.size() % 2 == 0);
 int half_size = mlndx_.size() / 2;
 if(indeed && half_size > 0){
  for(int i = 1; i < half_size; ++i){
   if(((bases_[i] + mlndx_[i]) < (bases_[i-1] + mlndx_[i-1])) ||
      ((bases_[i] + mlndx_[i]) != (bases_[half_size+i] + mlndx_[half_size+i]))){
    indeed = false;
    break;
   }
  }
  indeed = indeed && ((bases_[0] + mlndx_[0]) == (bases_[half_size] + mlndx_[half_size]));
 }
 return indeed;
}


inline bool TensorRange::nonincreasingOrderDiag() const
{
 bool indeed = (mlndx_.size() % 2 == 0);
 int half_size = mlndx_.size() / 2;
 if(indeed && half_size > 0){
  for(int i = 1; i < half_size; ++i){
   if(((bases_[i] + mlndx_[i]) > (bases_[i-1] + mlndx_[i-1])) ||
      ((bases_[i] + mlndx_[i]) != (bases_[half_size+i] + mlndx_[half_size+i]))){
    indeed = false;
    break;
   }
  }
  indeed = indeed && ((bases_[0] + mlndx_[0]) == (bases_[half_size] + mlndx_[half_size]));
 }
 return indeed;
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
   if(subrange_end_ > 0){ //subranging is set
    if(localOffset() >= subrange_end_) break;
   }
   if(--increment == 0) return true;
   i = 0;
  }else{
   mlndx_[i++] = 0;
  }
 }
 reset();
 return false;
}


inline bool TensorRange::prev(DimOffset increment)
{
 const int rank = mlndx_.size();
 int i = 0;
 while(i < rank){
  if(mlndx_[i] > 0){
   mlndx_[i]--;
   if(subrange_begin_ < volume_){
    if(localOffset() < subrange_begin_) break;
   }
   if(--increment == 0) return true;
   i = 0;
  }else{
   mlndx_[i++] = extents_[i] - 1;
  }
 }
 reset();
 return false;
}


inline bool TensorRange::shift(long long offset)
{
 long long local_offset = localOffset();
 long long range_begin = 0;
 long long range_end = volume_;
 if(subrange_end_ > 0){
  range_begin = subrange_begin_;
  range_end = subrange_end_;
 }
 local_offset += offset;
 if(local_offset >= range_begin && local_offset < range_end){
  for(unsigned int i = 0; i < extents_.size(); ++i){
   mlndx_[i] = local_offset % extents_[i];
   local_offset /= extents_[i];
  }
 }else{
  reset();
  return false;
 }
 return true;
}

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_RANGE_HPP_
