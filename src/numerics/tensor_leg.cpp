/** ExaTN::Numerics: Tensor leg (connection)
REVISION: 2019/11/12

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

TensorLeg::TensorLeg():
 TensorLeg(0,0)
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

bool tensorLegsAreCongruent(const std::vector<TensorLeg> * legs0,
                            const std::vector<TensorLeg> * legs1)
{
 if(legs0->size() != legs1->size()) return false;
 auto iter1 = legs1->cbegin();
 for(auto iter0 = legs0->cbegin(); iter0 != legs0->cend(); ++iter0, ++iter1){
  if(iter0->getDirection() != iter1->getDirection()) return false;
 }
 return true;
}

} //namespace numerics

} //namespace exatn
