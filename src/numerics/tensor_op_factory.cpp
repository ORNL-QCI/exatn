/** ExaTN::Numerics: Tensor operation factory
REVISION: 2020/04/13

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_op_factory.hpp"

namespace exatn{

namespace numerics{

TensorOpFactory::TensorOpFactory()
{
 registerTensorOp(TensorOpCode::CREATE,&TensorOpCreate::createNew);
 registerTensorOp(TensorOpCode::DESTROY,&TensorOpDestroy::createNew);
 registerTensorOp(TensorOpCode::TRANSFORM,&TensorOpTransform::createNew);
 registerTensorOp(TensorOpCode::SLICE,&TensorOpSlice::createNew);
 registerTensorOp(TensorOpCode::INSERT,&TensorOpInsert::createNew);
 registerTensorOp(TensorOpCode::ADD,&TensorOpAdd::createNew);
 registerTensorOp(TensorOpCode::CONTRACT,&TensorOpContract::createNew);
 registerTensorOp(TensorOpCode::DECOMPOSE_SVD3,&TensorOpDecomposeSVD3::createNew);
 registerTensorOp(TensorOpCode::DECOMPOSE_SVD2,&TensorOpDecomposeSVD2::createNew);
 registerTensorOp(TensorOpCode::ORTHOGONALIZE_SVD,&TensorOpOrthogonalizeSVD::createNew);
 registerTensorOp(TensorOpCode::ORTHOGONALIZE_MGS,&TensorOpOrthogonalizeMGS::createNew);
 registerTensorOp(TensorOpCode::BROADCAST,&TensorOpBroadcast::createNew);
 registerTensorOp(TensorOpCode::ALLREDUCE,&TensorOpAllreduce::createNew);
}

void TensorOpFactory::registerTensorOp(TensorOpCode opcode, createTensorOpFn creator)
{
 factory_map_[opcode] = creator;
 return;
}

std::unique_ptr<TensorOperation> TensorOpFactory::createTensorOp(TensorOpCode opcode)
{
 auto it = factory_map_.find(opcode);
 if(it != factory_map_.end()) return (it->second)();
 return std::unique_ptr<TensorOperation>(nullptr);
}

std::shared_ptr<TensorOperation> TensorOpFactory::createTensorOpShared(TensorOpCode opcode)
{
 return std::move(createTensorOp(opcode));
}

TensorOpFactory * TensorOpFactory::get()
{
 static TensorOpFactory single_instance;
 return &single_instance;
}

} //namespace numerics

} //namespace exatn
