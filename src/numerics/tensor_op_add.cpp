/** ExaTN::Numerics: Tensor operation: Adds a tensor to another tensor
REVISION: 2019/05/30

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_op_add.hpp"

namespace exatn{

namespace numerics{

TensorOpAdd::TensorOpAdd():
 TensorOperation(2,1)
{
 this->setScalar(0,std::complex<double>{1.0,0.0}); //default alpha prefactor
}

bool TensorOpAdd::isSet() const
{
 return (this->getNumOperandsSet() == this->getNumOperands() && this->getIndexPattern().length() > 0);
}

} //namespace numerics

} //namespace exatn
