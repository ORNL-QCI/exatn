/** ExaTN::Numerics: Tensor operation: Contracts two tensors and accumulates the result into another tensor
REVISION: 2019/05/30

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_op_contract.hpp"

namespace exatn{

namespace numerics{

TensorOpContract::TensorOpContract():
 TensorOperation(3,2)
{
 this->setScalar(0,std::complex<double>{1.0,0.0}); //default alpha prefactor
 this->setScalar(1,std::complex<double>{1.0,0.0}); //default beta prefactor (accumulative)
}

bool TensorOpContract::isSet() const
{
 return (this->getNumOperandsSet() == this->getNumOperands() && this->getIndexPattern().length() > 0);
}

} //namespace numerics

} //namespace exatn
