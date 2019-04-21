/** ExaTN::Numerics: Tensor network
REVISION: 2019/04/20

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:

**/

#ifndef TENSOR_NETWORK_HPP_
#define TENSOR_NETWORK_HPP_

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
 TensorNetwork(TensorNetwork &&) = default;
 TensorNetwork & operator=(TensorNetwork &&) = default;
 virtual ~TensorNetwork() = default;

private:

 std::unordered_map<unsigned int, TensorConn> tensors_; //tensors connected to each other via legs (tensor connections)

};

} //namespace numerics

} //namespace exatn

#endif //TENSOR_NETWORK_HPP_
