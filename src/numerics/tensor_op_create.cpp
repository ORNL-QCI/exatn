/** ExaTN::Numerics: Tensor operation: Creates a tensor
REVISION: 2019/08/28

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "exatn_service.hpp"

#include "tensor_op_create.hpp"

#include "tensor_node_executor.hpp"

namespace exatn{

namespace numerics{

TensorOpCreate::TensorOpCreate():
 TensorOperation(TensorOpCode::CREATE,1,0), element_type_(TensorElementType::REAL64)
{
}

bool TensorOpCreate::isSet() const
{
 return (this->getNumOperandsSet() == this->getNumOperands());
}

void TensorOpCreate::accept(runtime::TensorNodeExecutor & node_executor)
{
 node_executor.execute(*this);
 return;
}

std::unique_ptr<TensorOperation> TensorOpCreate::createNew()
{
 return std::unique_ptr<TensorOperation>(new TensorOpCreate());
}

void TensorOpCreate::resetTensorElementType(TensorElementType element_type)
{
 element_type_ = element_type;
 return;
}

} //namespace numerics

} //namespace exatn
