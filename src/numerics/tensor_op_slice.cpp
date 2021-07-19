/** ExaTN::Numerics: Tensor operation: Extracts a slice from a tensor
REVISION: 2021/07/15

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "exatn_service.hpp"

#include "tensor_op_slice.hpp"

#include "tensor_node_executor.hpp"

namespace exatn{

namespace numerics{

TensorOpSlice::TensorOpSlice():
 TensorOperation(TensorOpCode::SLICE,2,0,1+0*2,{0,1})
{
}

bool TensorOpSlice::isSet() const
{
 return (this->getNumOperandsSet() == this->getNumOperands());
}

int TensorOpSlice::accept(runtime::TensorNodeExecutor & node_executor,
                          runtime::TensorOpExecHandle * exec_handle)
{
 return node_executor.execute(*this,exec_handle);
}

std::unique_ptr<TensorOperation> TensorOpSlice::createNew()
{
 return std::unique_ptr<TensorOperation>(new TensorOpSlice());
}

std::size_t TensorOpSlice::decompose(const TensorMapper & tensor_mapper)
{
 assert(false);
 //`Implement
 return 0;
}

} //namespace numerics

} //namespace exatn
