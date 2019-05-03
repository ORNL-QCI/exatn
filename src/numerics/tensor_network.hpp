/** ExaTN::Numerics: Tensor network
REVISION: 2019/05/02

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) A tensor network is a set of connected tensors.
 (b) A tensor network is always closed, which in some
     cases requires introducing an explicit output tensor
     collecting all open ends of the original tensor network.
 (c) Each tensor in a tensor network can be connected to
     other tensors in that tensor network via tensor legs.
 (d) Each tensor leg in a given tensor is associated with
     one of the tensor dimensions, one tensor leg per each
     tensor dimension.
 (e) A tensor leg can connect a given tensor with one or more
     other tensors in the same tensor network. Thus, tensor
     legs can be binary, ternary, etc.
**/

#ifndef EXATN_NUMERICS_TENSOR_NETWORK_HPP_
#define EXATN_NUMERICS_TENSOR_NETWORK_HPP_

#include "tensor_basic.hpp"
#include "tensor_connected.hpp"

#include <unordered_map>

namespace exatn{

namespace numerics{

class TensorNetwork{
public:

 TensorNetwork() = default;

 TensorNetwork(const TensorNetwork &) = default;
 TensorNetwork & operator=(const TensorNetwork &) = default;
 TensorNetwork(TensorNetwork &&) noexcept = default;
 TensorNetwork & operator=(TensorNetwork &&) noexcept = default;
 virtual ~TensorNetwork() = default;

private:

 std::unordered_map<unsigned int, TensorConn> tensors_; //tensors connected to each other via legs (tensor connections)

};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_NETWORK_HPP_
