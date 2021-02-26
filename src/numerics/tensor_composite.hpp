/** ExaTN::Numerics: Composite tensor
REVISION: 2021/02/26

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) A composite tensor is a tensor which is explicitly partitioned
     into subtensors. The list of subtensors does not have to be
     complete, that is, there can be missing subtensors.
 (b) The subtensors are produced by a recursive bisection of one
     or more tensor dimensions to a desired depth. The max depth
     can be different for different tensor dimensions.
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
#include <string>
#include <vector>
#include <memory>

#include "errors.hpp"

namespace exatn{

namespace numerics{

class TensorComposite: public Tensor{
public:

 template<typename... Args>
 TensorComposite(const std::vector<std::pair<unsigned int, unsigned int>> & split_dims, //in: split tensor dimensions: pair{Dimension,MaxDepth}
                 Args&&... args);                                                       //in: arguments for Tensor ctor

 TensorComposite(const TensorComposite & tensor) = default;
 TensorComposite & operator=(const TensorComposite & tensor) = default;
 TensorComposite(TensorComposite && tensor) noexcept = default;
 TensorComposite & operator=(TensorComposite && tensor) noexcept = default;
 virtual ~TensorComposite() = default;

 //virtual void pack(BytePacket & byte_packet) const override;
 //virtual void unpack(BytePacket & byte_packet) override;

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

 unsigned int num_bisections_;                                    //total number of bisections
 std::vector<std::pair<unsigned int, unsigned int>> bisect_bits_; //bisection bits: Bit position --> {Dimension,Depth}
};


//TEMPLATE DEFINITIONS:

template<typename... Args>
TensorComposite::TensorComposite(const std::vector<std::pair<unsigned int, unsigned int>> & split_dims,
                                 Args&&... args):
 Tensor(std::forward<Args>(args)...), split_dims_(split_dims)
{
 //`Finish
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
