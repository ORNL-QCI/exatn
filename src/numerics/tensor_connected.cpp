/** ExaTN::Numerics: Tensor connected to other tensors inside a tensor network
REVISION: 2019/06/03

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_connected.hpp"

#include <iostream>

namespace exatn{

namespace numerics{

TensorConn::TensorConn(std::shared_ptr<Tensor> tensor, unsigned int id, const std::vector<TensorLeg> & legs):
 tensor_(tensor), id_(id), legs_(legs)
{
}

void TensorConn::printIt() const
{
 std::cout << id_ << ": ";
 tensor_->printIt();
 std::cout << ": { ";
 for(const auto & leg: legs_) leg.printIt();
 std::cout << " }" << std::endl;
 return;
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
