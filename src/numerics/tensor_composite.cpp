/** ExaTN::Numerics: Composite tensor
REVISION: 2021/02/26

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_composite.hpp"

namespace exatn{

namespace numerics{

void TensorComposite::pack(BytePacket & byte_packet) const
{
 Tensor::pack(byte_packet);
 //`Pack derived class members
 return;
}


void TensorComposite::unpack(BytePacket & byte_packet)
{
 Tensor::unpack(byte_packet);
 //`Unpack derived class members
 return;
}


void TensorComposite::generateSubtensors(std::function<bool (const Tensor &)> tensor_predicate)
{
 //`Finish
 return;
}

} //namespace numerics

} //namespace exatn
