/** ExaTN::Numerics: Tensor operation: Destroys a tensor
REVISION: 2019/06/03

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_op_destroy.hpp"

namespace exatn{

namespace numerics{

TensorOpDestroy::TensorOpDestroy():
 TensorOperation(TensorOpCode::DESTROY,1,0)
{
}

bool TensorOpDestroy::isSet() const
{
 return (this->getNumOperandsSet() == this->getNumOperands());
}

std::unique_ptr<TensorOperation> TensorOpDestroy::createNew()
{
 return std::unique_ptr<TensorOperation>(new TensorOpDestroy());
}

} //namespace numerics

} //namespace exatn
