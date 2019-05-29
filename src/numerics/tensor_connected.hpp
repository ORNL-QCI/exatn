/** ExaTN::Numerics: Tensor connected to other tensors in a tensor network
REVISION: 2019/05/27

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) A tensor inside a tensor network is generally connected
     to other tensors in that network via so-called tensor legs,
     each tensor leg is associated with its own tensor dimension.
 (b) Each tensor leg specifies a connection of a given tensor dimension
     to some dimension (or dimensions) in another tensor (or tensors) in
     the same tensor network. Thus, tensor legs can be binary, ternary, etc.,
     based on whether the tensor network is a graph or a hyper-graph.
 (c) The abstraction of a connected tensor is introduced for a quick
     inspection of the neighborhood of a chosen tensor inside the tensor network.
**/

#ifndef EXATN_NUMERICS_TENSOR_CONNECTED_HPP_
#define EXATN_NUMERICS_TENSOR_CONNECTED_HPP_

#include "tensor_basic.hpp"
#include "tensor_leg.hpp"
#include "tensor.hpp"

#include <vector>

namespace exatn{

namespace numerics{

class TensorConn{
public:

 /** Constructs a connected tensor inside a tensor network. **/
 TensorConn(const Tensor * tensor,                //non-owning pointer to the tensor
            unsigned int id,                      //tensor id in the tensor network
            const std::vector<TensorLeg> & legs); //tensor legs: Connections to other tensors in the tensor network

 TensorConn(const TensorConn &) = default;
 TensorConn & operator=(const TensorConn &) = default;
 TensorConn(TensorConn &&) noexcept = default;
 TensorConn & operator=(TensorConn &&) noexcept = default;
 virtual ~TensorConn() = default;

 /** Returns a non-owning pointer to the tensor. **/
 const Tensor * getTensor() const;

 /** Returns the tensor id. **/
 unsigned int getTensorId() const;

 /** Returns a specific tensor leg. **/
 TensorLeg getTensorLeg(unsigned int leg_id) const;

 /** Returns all tensor legs. **/
 const std::vector<TensorLeg> & getTensorLegs() const;

private:

 const Tensor * tensor_;       //non-owning pointer to the tensor
 unsigned int id_;             //tensor id in the tensor network
 std::vector<TensorLeg> legs_; //tensor legs: Connections to other tensors

};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_CONNECTED_HPP_
