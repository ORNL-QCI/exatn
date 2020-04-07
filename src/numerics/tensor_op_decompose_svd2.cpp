/** ExaTN::Numerics: Tensor operation: Decomposes a tensor into two tensor factors via SVD
REVISION: 2020/04/07

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "exatn_service.hpp"

#include "tensor_op_decompose_svd2.hpp"

#include "tensor_node_executor.hpp"

namespace exatn{

namespace numerics{

TensorOpDecomposeSVD2::TensorOpDecomposeSVD2():
 TensorOperation(TensorOpCode::DECOMPOSE_SVD2,3,0,1+1*2+0*4)
{
}

bool TensorOpDecomposeSVD2::isSet() const
{
 return (this->getNumOperandsSet() == this->getNumOperands() && this->getIndexPattern().length() > 0);
}

int TensorOpDecomposeSVD2::accept(runtime::TensorNodeExecutor & node_executor,
                                  runtime::TensorOpExecHandle * exec_handle)
{
 return node_executor.execute(*this,exec_handle);
}

std::unique_ptr<TensorOperation> TensorOpDecomposeSVD2::createNew()
{
 return std::unique_ptr<TensorOperation>(new TensorOpDecomposeSVD2());
}

} //namespace numerics

} //namespace exatn
