/** ExaTN::Numerics: Tensor network
REVISION: 2019/05/31

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) A tensor network is a set of connected tensors.
 (b) Each tensor in a tensor network can be connected to
     other tensors in that tensor network via tensor legs.
 (c) Each tensor leg in a given tensor is uniquely associated
     with one of its dimensions, one tensor leg per tensor dimension.
 (d) A tensor leg can connect a given tensor with one or more
     other tensors in the same tensor network. Thus, tensor
     legs can be binary, ternary, etc.
 (e) A tensor network is always closed, which in some
     cases requires introducing an explicit output tensor
     collecting all open ends of the original tensor network.
**/

#ifndef EXATN_NUMERICS_TENSOR_NETWORK_HPP_
#define EXATN_NUMERICS_TENSOR_NETWORK_HPP_

#include "tensor_basic.hpp"
#include "tensor_connected.hpp"
#include "tensor_op_factory.hpp"

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
                                                        //map: Nonnegative tensor id --> Connected tensor

};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_NETWORK_HPP_
