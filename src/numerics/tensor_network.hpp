/** ExaTN::Numerics: Tensor network
REVISION: 2019/07/02

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
 (e) A tensor network is always closed, which requires introducing
     an explicit output tensor collecting all open legs of the original
     tensor network. If the original tensor network does not have open
     legs, the output tensor is simply a scalar which the original tensor
     network evaluates to; otherwise, a tensor network evaluates to a tensor.
 (f) Tensor enumeration:
     0: Output tensor/scalar which the tensor network evaluates to;
     1..N: Input tensors/scalars constituting the original tensor network;
     N+1..M: Intermediate tensors obtained by contractions of input tensors.
**/

#ifndef EXATN_NUMERICS_TENSOR_NETWORK_HPP_
#define EXATN_NUMERICS_TENSOR_NETWORK_HPP_

#include "tensor_basic.hpp"
#include "tensor_connected.hpp"
#include "tensor_op_factory.hpp"

#include <unordered_map>
#include <string>

namespace exatn{

namespace numerics{

class TensorNetwork{
public:

 static constexpr unsigned int NUM_WALKERS = 1024; //default number of walkers for tensor contraction sequence optimization

 using ContractionSequence = std::vector<std::pair<unsigned int, unsigned int>>; //pairs of contracted tensor id's

 /** Creates an unnamed empty tensor network with a single scalar output tensor named "_SMOKY_TENSOR_" **/
 TensorNetwork();
 /** Creates a named empty tensor network with a single scalar output tensor named with the same name. **/
 TensorNetwork(const std::string & name);

 TensorNetwork(const TensorNetwork &) = default;
 TensorNetwork & operator=(const TensorNetwork &) = default;
 TensorNetwork(TensorNetwork &&) noexcept = default;
 TensorNetwork & operator=(TensorNetwork &&) noexcept = default;
 virtual ~TensorNetwork() = default;

 /** Prints **/
 void printIt() const;

 /** Returns the number of input tensors in the tensor network.
     Note that the output tensor (tensor #0) is not counted here. **/
 unsigned int getNumTensors() const;

private:

 std::string name_;                                     //tensor network name
 std::unordered_map<unsigned int, TensorConn> tensors_; //tensors connected to each other via legs (tensor connections)
                                                        //map: Non-negative tensor id --> Connected tensor
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_NETWORK_HPP_
