/** ExaTN::Numerics: Tensor operation: Destroys a tensor
REVISION: 2020/04/07

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "exatn_service.hpp"

#include "tensor_op_destroy.hpp"

#include "tensor_node_executor.hpp"

namespace exatn{

namespace numerics{

TensorOpDestroy::TensorOpDestroy():
 TensorOperation(TensorOpCode::DESTROY,1,0,1)
{
}

bool TensorOpDestroy::isSet() const
{
 return (this->getNumOperandsSet() == this->getNumOperands());
}

int TensorOpDestroy::accept(runtime::TensorNodeExecutor & node_executor,
                            runtime::TensorOpExecHandle * exec_handle)
{
 return node_executor.execute(*this,exec_handle);
}

std::unique_ptr<TensorOperation> TensorOpDestroy::createNew()
{
 return std::unique_ptr<TensorOperation>(new TensorOpDestroy());
}

} //namespace numerics

} //namespace exatn
