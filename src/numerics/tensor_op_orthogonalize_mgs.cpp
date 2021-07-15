/** ExaTN::Numerics: Tensor operation: Orthogonalizes a tensor via MGS
REVISION: 2021/07/15

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "exatn_service.hpp"

#include "tensor_op_orthogonalize_mgs.hpp"

#include "tensor_node_executor.hpp"

namespace exatn{

namespace numerics{

TensorOpOrthogonalizeMGS::TensorOpOrthogonalizeMGS():
 TensorOperation(TensorOpCode::ORTHOGONALIZE_MGS,1,0,1,{0})
{
}

bool TensorOpOrthogonalizeMGS::isSet() const
{
 return (this->getNumOperandsSet() == this->getNumOperands() && this->getIndexPattern().length() > 0);
}

int TensorOpOrthogonalizeMGS::accept(runtime::TensorNodeExecutor & node_executor,
                                     runtime::TensorOpExecHandle * exec_handle)
{
 return node_executor.execute(*this,exec_handle);
}

std::unique_ptr<TensorOperation> TensorOpOrthogonalizeMGS::createNew()
{
 return std::unique_ptr<TensorOperation>(new TensorOpOrthogonalizeMGS());
}

std::size_t TensorOpOrthogonalizeMGS::decompose(const TensorMapper & tensor_mapper)
{
 assert(false);
 //`Implement
 return 0;
}

} //namespace numerics

} //namespace exatn
