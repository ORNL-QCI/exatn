/** ExaTN::Numerics: Tensor leg (connection)
REVISION: 2019/10/16

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_leg.hpp"

namespace exatn{

namespace numerics{

TensorLeg::TensorLeg(unsigned int tensor_id,
                     unsigned int dimensn_id,
                     LegDirection direction):
tensor_id_(tensor_id), dimensn_id_(dimensn_id), direction_(direction)
{
}

void TensorLeg::printIt() const
{
 if(direction_ == LegDirection::INWARD){
  std::cout << "{" << tensor_id_ << ":" << dimensn_id_ << ";+}";
 }else if(direction_ == LegDirection::OUTWARD){
  std::cout << "{" << tensor_id_ << ":" << dimensn_id_ << ";-}";
 }else{
  std::cout << "{" << tensor_id_ << ":" << dimensn_id_ << "}";
 }
 return;
}

unsigned int TensorLeg::getTensorId() const
{
 return tensor_id_;
}

unsigned int TensorLeg::getDimensionId() const
{
 return dimensn_id_;
}

LegDirection TensorLeg::getDirection() const
{
 return direction_;
}

void TensorLeg::resetConnection(unsigned int tensor_id,
                                unsigned int dimensn_id,
                                LegDirection direction)
{
 tensor_id_ = tensor_id;
 dimensn_id_ = dimensn_id;
 direction_ = direction;
 return;
}

void TensorLeg::resetTensorId(unsigned int tensor_id)
{
 tensor_id_ = tensor_id;
 return;
}

void TensorLeg::resetDimensionId(unsigned int dimensn_id)
{
 dimensn_id_ = dimensn_id;
 return;
}

void TensorLeg::resetDirection(LegDirection direction)
{
 direction_ = direction;
 return;
}

void TensorLeg::reverseDirection()
{
 direction_ = reverseLegDirection(direction_);
 return;
}

} //namespace numerics

} //namespace exatn
