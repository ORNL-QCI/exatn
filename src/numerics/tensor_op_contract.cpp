/** ExaTN::Numerics: Tensor operation: Contracts two tensors and accumulates the result into another tensor
REVISION: 2020/06/06

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "exatn_service.hpp"

#include "tensor_op_contract.hpp"

#include "tensor_node_executor.hpp"

#include <cmath>

namespace exatn{

namespace numerics{

TensorOpContract::TensorOpContract():
 TensorOperation(TensorOpCode::CONTRACT,3,2,1+0*2+0*4)
{
 this->setScalar(0,std::complex<double>{1.0,0.0}); //default alpha prefactor
 this->setScalar(1,std::complex<double>{1.0,0.0}); //default beta prefactor (accumulative)
}

bool TensorOpContract::isSet() const
{
 return (this->getNumOperandsSet() == this->getNumOperands() && this->getIndexPattern().length() > 0);
}

int TensorOpContract::accept(runtime::TensorNodeExecutor & node_executor,
                             runtime::TensorOpExecHandle * exec_handle)
{
 return node_executor.execute(*this,exec_handle);
}

double TensorOpContract::getFlopEstimate() const
{
 if(this->isSet()){
  double vol0 = static_cast<double>(this->getTensorOperand(0)->getVolume());
  double vol1 = static_cast<double>(this->getTensorOperand(1)->getVolume());
  double vol2 = static_cast<double>(this->getTensorOperand(2)->getVolume());
  return std::sqrt(vol0*vol1*vol2)*2.0;
 }
 return 0.0;
}

std::unique_ptr<TensorOperation> TensorOpContract::createNew()
{
 return std::unique_ptr<TensorOperation>(new TensorOpContract());
}

} //namespace numerics

} //namespace exatn
