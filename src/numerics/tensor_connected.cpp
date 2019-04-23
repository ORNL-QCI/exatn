/** ExaTN::Numerics: Tensor connected to other tensors
REVISION: 2019/04/22

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_connected.hpp"

namespace exatn{

namespace numerics{

TensorConn::TensorConn(const Tensor * tensor, unsigned int id, const std::vector<TensorLeg> & legs):
 tensor_(tensor), id_(id), legs_(legs)
{
}

} //namespace numerics

} //namespace exatn
