/** ExaTN::Numerics: Tensor connected to other tensors
REVISION: 2019/04/20

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

 TensorConn() = default;

 TensorConn(const TensorConn &) = default;
 TensorConn & operator=(const TensorConn &) = default;
 TensorConn(TensorConn &&) = default;
 TensorConn & operator=(TensorConn &&) = default;
 virtual ~TensorConn() = default;

private:

 Tensor * tensor_;             //non-owning pointer to the tensor
 std::vector<TensorLeg> legs_; //tensor legs: Connections to other tensors

};

} //namespace numerics

} //namespace exatn

#endif //TENSOR_CONNECTED_HPP_
