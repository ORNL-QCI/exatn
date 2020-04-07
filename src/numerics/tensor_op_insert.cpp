/** ExaTN::Numerics: Tensor operation: Inserts a slice into a tensor
REVISION: 2020/04/07

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "exatn_service.hpp"

#include "tensor_op_insert.hpp"

#include "tensor_node_executor.hpp"

namespace exatn{

namespace numerics{

TensorOpInsert::TensorOpInsert():
 TensorOperation(TensorOpCode::INSERT,2,0,1+0*2)
{
}

bool TensorOpInsert::isSet() const
{
 return (this->getNumOperandsSet() == this->getNumOperands());
}

int TensorOpInsert::accept(runtime::TensorNodeExecutor & node_executor,
                           runtime::TensorOpExecHandle * exec_handle)
{
 return node_executor.execute(*this,exec_handle);
}

std::unique_ptr<TensorOperation> TensorOpInsert::createNew()
{
 return std::unique_ptr<TensorOperation>(new TensorOpInsert());
}

} //namespace numerics

} //namespace exatn
