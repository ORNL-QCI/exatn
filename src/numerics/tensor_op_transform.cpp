/** ExaTN::Numerics: Tensor operation: Transforms/initializes a tensor
REVISION: 2019/08/15

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "exatn_service.hpp"

#include "tensor_op_transform.hpp"

#include "tensor_node_executor.hpp"

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

void TensorOpTransform::accept(runtime::TensorNodeExecutor & node_executor)
{
 node_executor.execute(*this);
 return;
}

std::unique_ptr<TensorOperation> TensorOpTransform::createNew()
{
 return std::unique_ptr<TensorOperation>(new TensorOpTransform());
}

} //namespace numerics

} //namespace exatn
