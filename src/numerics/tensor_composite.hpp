/** ExaTN::Numerics: Composite tensor
REVISION: 2021/02/26

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) A composite tensor is a tensor which is explicitly partitioned
     into subtensors. The list of subtensors does not have to be
     complete, that is, there can be missing subtensors in general.
 (b) The subtensors are produced by a recursive bisection of one
     or more tensor dimensions to a desired maximal depth. The maximal
     depth can be different for different tensor dimensions.
     Each subtensor is thus identified by a bit sequence of length
     equal to the total number of bisections along all tensor dimensions,
     where each bit selects the left/right half in each bisection.
     The bits are additionally ordered by the bisection depth:
      bits for depth 0, bits for depth 1, ..., bits for depth MAX.
**/

#ifndef EXATN_NUMERICS_TENSOR_COMPOSITE_HPP_
#define EXATN_NUMERICS_TENSOR_COMPOSITE_HPP_

#include "tensor_basic.hpp"
#include "tensor.hpp"

#include <unordered_map>
#include <functional>
#include <algorithm>
#include <string>
#include <vector>
#include <memory>

#include "errors.hpp"

namespace exatn{

namespace numerics{

class TensorComposite : public Tensor{
public:

 /** Constructs a composite tensor by activating bisection on given tensor dimensions
     up to the provided maximal depth (max depth of 0 means no bisection). Note that
     the order of tensor dimensions matters as the produced subtensors are ordered
     because their producing bisections are ordered. If the provided values of MaxDepth
     are zero for all given tensor dimensions, no bisection will be done, resulting
     in a composite tensor consisting of itself (a single subtensor = tensor). **/
 template<typename... Args>
 TensorComposite(std::function<bool (const Tensor &)> tensor_predicate,                 //in: tensor predicate deciding which subtensors will exist
                 const std::vector<std::pair<unsigned int, unsigned int>> & split_dims, //in: split tensor dimensions: pair{Dimension,MaxDepth}
                 Args&&... args);                                                       //in: arguments for base Tensor ctor

 template<typename... Args>
 TensorComposite(const std::vector<std::pair<unsigned int, unsigned int>> & split_dims, //in: split tensor dimensions: pair{Dimension,MaxDepth}
                 Args&&... args);                                                       //in: arguments for base Tensor ctor

 TensorComposite(const TensorComposite & tensor) = default;
 TensorComposite & operator=(const TensorComposite & tensor) = default;
 TensorComposite(TensorComposite && tensor) noexcept = default;
 TensorComposite & operator=(TensorComposite && tensor) noexcept = default;
 virtual ~TensorComposite() = default;

 virtual void pack(BytePacket & byte_packet) const override;
 virtual void unpack(BytePacket & byte_packet) override;

 /** Returns a subtensor associated with a given integer id (bit-string),
     or nullptr if no such subtensor exists. **/
 inline std::shared_ptr<Tensor> operator[](unsigned long long subtensor_bits);

 /** Returns the total number of possible subtensors (in a complete set),
     given the number of bisections used in the constructor. **/
 inline unsigned long long getNumSubtensorsComplete() const;

 /** Returns the actual number of subtensors. **/
 inline unsigned long long getNumSubtensors() const;

 /** Returns the total number of bisections. **/
 inline unsigned int getNumBisections() const;

 /** Returns the position of the bit corresponding to a given tensor
     dimension bisected at a given depth. **/
 inline unsigned int bitPosition(unsigned int tensor_dimension,
                                 unsigned int depth) const;

 /** Returns the tensor dimension and bisection depth corresponding
     to a given bit position. **/
 inline std::pair<unsigned int, unsigned int> dimPositionAndDepth(unsigned int bit_position) const;

protected:

 const std::vector<std::pair<unsigned int, unsigned int>> split_dims_; //split tensor dimensions: pair{Dimension,MaxDepth}
 std::unordered_map<unsigned long long, std::shared_ptr<Tensor>> subtensors_; //subtensors identified by their bit-strings

private:

 void generateSubtensors(std::function<bool (const Tensor &)> tensor_predicate); //in: tensor predicate deciding which subtensors will exist

 unsigned int num_bisections_;                                    //total number of bisections
 std::vector<std::pair<unsigned int, unsigned int>> bisect_bits_; //bisection bits: Bit position --> {Dimension,Depth}
};


//TEMPLATE DEFINITIONS:

template<typename... Args>
TensorComposite::TensorComposite(std::function<bool (const Tensor &)> tensor_predicate,
                                 const std::vector<std::pair<unsigned int, unsigned int>> & split_dims,
                                 Args&&... args):
 Tensor(std::forward<Args>(args)...), split_dims_(split_dims)
{
 num_bisections_ = 0;
 auto tensor_rank = getRank();
 for(const auto & split_dim: split_dims_){
  assert(split_dim.first < tensor_rank);
  num_bisections_ += split_dim.second;
 }
 bisect_bits_.resize(num_bisections_);
 if(num_bisections_ > 0){
  unsigned int n = 0;
  for(const auto & split_dim: split_dims_){
   for(unsigned int depth = 1; depth <= split_dim.second; ++depth){
    bisect_bits_[n++]={split_dim.first,depth};
   }
  }
  if(num_bisections_ > 1){
   std::stable_sort(bisect_bits_.begin(),bisect_bits_.end(),
                    [](const std::pair<unsigned int, unsigned int> & a,
                       const std::pair<unsigned int, unsigned int> & b){
                     return (a.second < b.second);
                    });
  }
  generateSubtensors(tensor_predicate);
 }else{ //no bisections: Store itself as the base Tensor
  //`How to do it right?
 }
}


template<typename... Args>
TensorComposite::TensorComposite(const std::vector<std::pair<unsigned int, unsigned int>> & split_dims,
                                 Args&&... args):
 TensorComposite([](const Tensor &){return true;},split_dims,std::forward<Args>(args)...)
{
}


inline std::shared_ptr<Tensor> TensorComposite::operator[](unsigned long long subtensor_bits)
{
 assert(subtensor_bits < getNumSubtensorsComplete());
 auto iter = subtensors_.find(subtensor_bits);
 if(iter != subtensors_.end()) return iter->second;
 return std::shared_ptr<Tensor>(nullptr);
}


inline unsigned long long TensorComposite::getNumSubtensorsComplete() const
{
 return (1ULL << num_bisections_);
}


inline unsigned long long TensorComposite::getNumSubtensors() const
{
 return subtensors_.size();
}


inline unsigned int TensorComposite::getNumBisections() const
{
 return num_bisections_;
}


inline unsigned int TensorComposite::bitPosition(unsigned int tensor_dimension,
                                                 unsigned int depth) const
{
 //`Finish
 return 0;
}


inline std::pair<unsigned int, unsigned int> TensorComposite::dimPositionAndDepth(unsigned int bit_position) const
{
 unsigned int tensor_dim, dim_depth;
 //`Finish
 return std::make_pair(tensor_dim,dim_depth);
}

} //namespace numerics


/** Creates a new TensorComposite as a shared pointer to the base Tensor. **/
template<typename... Args>
inline std::shared_ptr<numerics::Tensor> makeSharedTensorComposite(Args&&... args)
{
 return std::shared_ptr<numerics::Tensor>{new numerics::TensorComposite(std::forward<Args>(args)...)};
}

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_COMPOSITE_HPP_
