/** ExaTN::Numerics: Tensor connected to other tensors
REVISION: 2019/04/22

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:

**/

#ifndef TENSOR_CONNECTED_HPP_
#define TENSOR_CONNECTED_HPP_

#include "tensor_basic.hpp"
#include "tensor_leg.hpp"
#include "tensor.hpp"

#include <vector>

namespace exatn{

namespace numerics{

class TensorConn{
public:

 TensorConn(const Tensor * tensor,
            unsigned int id,
            const std::vector<TensorLeg> & legs);

 TensorConn(const TensorConn &) = default;
 TensorConn & operator=(const TensorConn &) = default;
 TensorConn(TensorConn &&) = default;
 TensorConn & operator=(TensorConn &&) = default;
 virtual ~TensorConn() = default;

private:

 const Tensor * tensor_;       //non-owning pointer to the tensor
 unsigned int id_;             //tensor id in the tensor network
 std::vector<TensorLeg> legs_; //tensor legs: Connections to other tensors

};

} //namespace numerics

} //namespace exatn

#endif //TENSOR_CONNECTED_HPP_
