/** ExaTN::Numerics: Tensor leg (connection)
REVISION: 2018/10/31

Copyright (C) 2018-2018 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2018 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_leg.hpp"

namespace exatn{

namespace numerics{

TensorLeg::TensorLeg(unsigned int tensor_id,
                     unsigned int dimensn_id):
tensor_id_(tensor_id), dimensn_id_(dimensn_id)
{
}

void TensorLeg::printIt() const
{
 std::cout << "{" << tensor_id_ << ":" << dimensn_id_ << "}";
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

void TensorLeg::resetConnection(unsigned int tensor_id,
                                unsigned int dimensn_id)
{
 tensor_id_=tensor_id; dimensn_id_=dimensn_id;
 return;
}

void TensorLeg::resetTensorId(unsigned int tensor_id)
{
 tensor_id_=tensor_id;
 return;
}

void TensorLeg::resetDimensionId(unsigned int dimensn_id)
{
 dimensn_id_=dimensn_id;
 return;
}

} //namespace numerics

} //namespace exatn
