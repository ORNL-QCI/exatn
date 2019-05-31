/** ExaTN::Numerics: Tensor connected to other tensors inside a tensor network
REVISION: 2019/05/31

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_connected.hpp"

namespace exatn{

namespace numerics{

TensorConn::TensorConn(std::shared_ptr<Tensor> tensor, unsigned int id, const std::vector<TensorLeg> & legs):
 tensor_(tensor), id_(id), legs_(legs)
{
}

std::shared_ptr<Tensor> TensorConn::getTensor()
{
 return tensor_;
}

unsigned int TensorConn::getTensorId() const
{
 return id_;
}

TensorLeg TensorConn::getTensorLeg(unsigned int leg_id) const
{
 assert(leg_id < legs_.size());
 return legs_[leg_id];
}

const std::vector<TensorLeg> & TensorConn::getTensorLegs() const
{
 return legs_;
}

} //namespace numerics

} //namespace exatn
