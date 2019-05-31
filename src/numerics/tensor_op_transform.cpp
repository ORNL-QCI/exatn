/** ExaTN::Numerics: Tensor operation: Transforms/initializes a tensor
REVISION: 2019/05/31

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_op_transform.hpp"

namespace exatn{

namespace numerics{

TensorOpTransform::TensorOpTransform():
 TensorOperation(TensorOpCode::TRANSFORM,1,1)
{
 this->setScalar(0,std::complex<double>{0.0,0.0}); //default numerical initialization value
}

bool TensorOpTransform::isSet() const
{
 return (this->getNumOperandsSet() == this->getNumOperands());
}

} //namespace numerics

} //namespace exatn
