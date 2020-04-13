/** ExaTN::Numerics: Tensor operation: Decomposes a tensor into three tensor factors via SVD
REVISION: 2020/04/13

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "exatn_service.hpp"

#include "tensor_op_decompose_svd3.hpp"

#include "tensor_node_executor.hpp"

namespace exatn{

namespace numerics{

TensorOpDecomposeSVD3::TensorOpDecomposeSVD3():
 TensorOperation(TensorOpCode::DECOMPOSE_SVD3,4,0,1+1*2+1*4+0*8),
 absorb_singular_values_('N')
{
}

bool TensorOpDecomposeSVD3::isSet() const
{
 return (this->getNumOperandsSet() == this->getNumOperands() && this->getIndexPattern().length() > 0);
}

int TensorOpDecomposeSVD3::accept(runtime::TensorNodeExecutor & node_executor,
                                  runtime::TensorOpExecHandle * exec_handle)
{
 return node_executor.execute(*this,exec_handle);
}

std::unique_ptr<TensorOperation> TensorOpDecomposeSVD3::createNew()
{
 return std::unique_ptr<TensorOperation>(new TensorOpDecomposeSVD3());
}

bool TensorOpDecomposeSVD3::resetAbsorptionMode(const char absorb_mode)
{
 if(absorb_mode == 'N' || absorb_mode == 'L' ||
    absorb_mode == 'R' || absorb_mode == 'S'){
  absorb_singular_values_ = absorb_mode;
 }else{
  return false;
 }
 return true;
}

} //namespace numerics

} //namespace exatn
