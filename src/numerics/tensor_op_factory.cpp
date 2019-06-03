/** ExaTN::Numerics: Tensor operation factory
REVISION: 2019/06/03

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_op_factory.hpp"

namespace exatn{

namespace numerics{

TensorOpFactory::TensorOpFactory()
{
 registerTensorOp(TensorOpCode::CREATE,&TensorOpCreate::createNew);
 registerTensorOp(TensorOpCode::DESTROY,&TensorOpDestroy::createNew);
 registerTensorOp(TensorOpCode::TRANSFORM,&TensorOpTransform::createNew);
 registerTensorOp(TensorOpCode::ADD,&TensorOpAdd::createNew);
 registerTensorOp(TensorOpCode::CONTRACT,&TensorOpContract::createNew);
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

TensorOpFactory * TensorOpFactory::get()
{
 static TensorOpFactory single_instance;
 return &single_instance;
}

} //namespace numerics

} //namespace exatn
