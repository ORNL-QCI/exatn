/** ExaTN::Numerics: Composite tensor
REVISION: 2021/03/07

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
     {bits for depth 1, bits for depth 2, ..., bits for depth MAX}.
     That is, the bit sequence identifying each subtensor looks like:
     {dims split at depth 1; dims split at depth 2; ...; dims split at max depth},
     where dimensions at each depth are ordered in the user-specified order.
     In general, dimensions at depth d+1 form a subset of dimensions at depth d.
     The produced subtensors are ordered with respect to their bit-strings.
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

 /** For iterating over subtensors. **/
 using Iterator = std::unordered_map<unsigned long long, std::shared_ptr<Tensor>>::iterator;
 using ConstIterator = std::unordered_map<unsigned long long, std::shared_ptr<Tensor>>::const_iterator;

 /** Constructs a composite tensor by activating bisection on given tensor dimensions
     up to the provided maximal depth (max depth of 0 means no bisection). Note that
     the order of tensor dimensions matters as the produced subtensors are ordered
     because their producing bisections are ordered. If the provided values of MaxDepth
     are zero for all given tensor dimensions no bisection will be done, resulting
     in a composite tensor consisting of itself (a single subtensor = tensor). Note
     that in this case the corresponding base tensor will be stored as a clone. **/
 template<typename... Args>
 TensorComposite(std::function<bool (const Tensor &)> tensor_predicate,                 //in: tensor predicate deciding which subtensors will exist
                 const std::vector<std::pair<unsigned int, unsigned int>> & split_dims, //in: split tensor dimensions: pair{Dimension,MaxDepth}
                 Args&&... args);                                                       //in: arguments for base Tensor ctor

 template<typename... Args>
 TensorComposite(const std::vector<std::pair<unsigned int, unsigned int>> & split_dims, //in: split tensor dimensions: pair{Dimension,MaxDepth}
                 Args&&... args);                                                       //in: arguments for base Tensor ctor

 TensorComposite(BytePacket & byte_packet);

 TensorComposite(const TensorComposite & tensor) = default;
 TensorComposite & operator=(const TensorComposite & tensor) = default;
 TensorComposite(TensorComposite && tensor) noexcept = default;
 TensorComposite & operator=(TensorComposite && tensor) noexcept = default;
 virtual ~TensorComposite() = default;

 virtual void pack(BytePacket & byte_packet) const override;
 virtual void unpack(BytePacket & byte_packet) override;

 virtual bool isComposite() const override;

 inline Iterator begin() {return subtensors_.begin();}
 inline Iterator end() {return subtensors_.end();}
 inline ConstIterator cbegin() {return subtensors_.cbegin();}
 inline ConstIterator cend() {return subtensors_.cend();}

 /** Returns a subtensor associated with a given bit-string (integer id),
     or nullptr if no such subtensor exists. **/
 inline std::shared_ptr<Tensor> operator[](unsigned long long subtensor_bits);

 /** Returns the total number of possible subtensors (in a complete set),
     given the number of bisections used in the constructor. **/
 inline unsigned long long getNumSubtensorsComplete() const;

 /** Returns the actual number of subtensors. **/
 inline unsigned long long getNumSubtensors() const;

 /** Returns the total number of bisections. **/
 inline unsigned int getNumBisections() const;

 /** Return the max depth of a given tensor dimension. **/
 inline unsigned int getDimDepth(unsigned int dimensn) const;

 /** Returns the position of the bit corresponding to a given tensor
     dimension bisected at a given depth. In case there is no such bit,
     returns the total number of bisections (= end of container). **/
 inline unsigned int bitPosition(unsigned int tensor_dimension,
                                 unsigned int depth) const;

 /** Returns the tensor dimension and bisection depth corresponding
     to a given bit position (= given bisection). **/
 inline std::pair<unsigned int, unsigned int> dimPositionAndDepth(unsigned int bit_position) const;

protected:

 std::vector<std::pair<unsigned int, unsigned int>> split_dims_; //split tensor dimensions: pair{Dimension,MaxDepth}
 std::unordered_map<unsigned long long, std::shared_ptr<Tensor>> subtensors_; //subtensors identified by their bit-strings

private:

 void packTensorComposite(BytePacket & byte_packet) const;
 void unpackTensorComposite(BytePacket & byte_packet);

 /** Generates subtensors filtered by a given tensor existence predicate. **/
 void generateSubtensors(std::function<bool (const Tensor &)> tensor_predicate); //in: tensor predicate deciding which subtensors will exist

 unsigned int num_bisections_;                                    //total number of bisections
 std::vector<std::pair<unsigned int, unsigned int>> bisect_bits_; //bisection bits: Bit position --> {Dimension,Depth}
 std::vector<unsigned int> dim_depth_;                            //tensor dimension depth (number of bisections): [0..max]
};


//TEMPLATE DEFINITIONS:

template<typename... Args>
TensorComposite::TensorComposite(std::function<bool (const Tensor &)> tensor_predicate,
                                 const std::vector<std::pair<unsigned int, unsigned int>> & split_dims,
                                 Args&&... args):
 Tensor(std::forward<Args>(args)...), split_dims_(split_dims)
{
 num_bisections_ = 0;
 auto tensor_rank = Tensor::getRank();
 dim_depth_.resize(tensor_rank,0);
 for(const auto & split_dim: split_dims_){
  assert(split_dim.first < tensor_rank);
  num_bisections_ += split_dim.second;
  dim_depth_[split_dim.first] = split_dim.second;
 }
 bisect_bits_.resize(num_bisections_);
 if(num_bisections_ > 0){ //generate subtensors
  unsigned int n = 0;
  for(const auto & split_dim: split_dims_){
   for(unsigned int depth = 1; depth <= split_dim.second; ++depth){
    bisect_bits_[n++]={split_dim.first,depth};
   }
  }
  if(num_bisections_ > 1){ //sort by depth
   std::stable_sort(bisect_bits_.begin(),bisect_bits_.end(),
                    [](const std::pair<unsigned int, unsigned int> & a,
                       const std::pair<unsigned int, unsigned int> & b){
                     return (a.second < b.second);
                    });
  }
  generateSubtensors(tensor_predicate);
 }else{ //no bisections: Store itself as a clone of the base Tensor
  auto res = subtensors_.emplace(std::make_pair(0ULL,std::make_shared<Tensor>(static_cast<Tensor>(*this))));
  assert(res.second);
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
 return (1ULL << num_bisections_); //2^num_bisections
}


inline unsigned long long TensorComposite::getNumSubtensors() const
{
 return subtensors_.size();
}


inline unsigned int TensorComposite::getNumBisections() const
{
 return num_bisections_;
}


inline unsigned int TensorComposite::getDimDepth(unsigned int dimensn) const
{
 assert(dimensn < getRank());
 return dim_depth_[dimensn];
}


inline unsigned int TensorComposite::bitPosition(unsigned int tensor_dimension,
                                                 unsigned int depth) const
{
 auto cmp_lt = [](const std::pair<unsigned int, unsigned int> & a,
                  const std::pair<unsigned int, unsigned int> & b){
                return (a.second < b.second || ((a.second == b.second) && (a.first < b.first)));
               };

 auto cmp_eq = [](const std::pair<unsigned int, unsigned int> & a,
                  const std::pair<unsigned int, unsigned int> & b){
                return ((a.second == b.second) && (a.first == b.first));
               };

 const auto target_pair = std::make_pair(tensor_dimension,depth);
 unsigned int left = 0, right = (num_bisections_ - 1);
 unsigned int pos = num_bisections_;
 if(cmp_eq(bisect_bits_[left],target_pair)){
  pos = left;
 }else if(cmp_eq(bisect_bits_[right],target_pair)){
  pos = right;
 }else{
  while(left < right){
   pos = (left + right) / 2;
   if(pos != left && pos != right){
    if(cmp_lt(bisect_bits_[pos],target_pair)){
     left = pos;
    }else if(cmp_lt(target_pair,bisect_bits_[pos])){
     right = pos;
    }else{
     break; //found
    }
   }else{
    pos = num_bisections_; //not found
    ++left; //done
   }
  }
 }
 return pos;
}


inline std::pair<unsigned int, unsigned int> TensorComposite::dimPositionAndDepth(unsigned int bit_position) const
{
 assert(bit_position < num_bisections_);
 return bisect_bits_[bit_position];
}

} //namespace numerics


/** Creates a new TensorComposite as a shared pointer to the base Tensor. **/
template<typename... Args>
inline std::shared_ptr<numerics::Tensor> makeSharedTensorComposite(Args&&... args)
{
 return std::shared_ptr<numerics::Tensor>{new numerics::TensorComposite(std::forward<Args>(args)...)};
}


/** Downcasts a Tensor as TensorComposite if it is TensorComposite, otherwise returns nullptr. **/
inline std::shared_ptr<numerics::TensorComposite> castTensorComposite(std::shared_ptr<numerics::Tensor> tensor)
{
 return std::dynamic_pointer_cast<numerics::TensorComposite>(tensor);
}

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_COMPOSITE_HPP_
