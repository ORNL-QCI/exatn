/** ExaTN::Numerics: Numerical server
REVISION: 2019/09/24

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "num_server.hpp"

#include "tensor_symbol.hpp"

#include <cassert>
#include <vector>
#include <map>

namespace exatn{

/** Numerical server (singleton) **/
std::shared_ptr<NumServer> numericalServer {nullptr}; //initialized by exatn::initialize()


NumServer::NumServer():
 tensor_rt_(std::make_shared<runtime::TensorRuntime>())
{
 tensor_op_factory_ = TensorOpFactory::get();
 scopes_.push(std::pair<std::string,ScopeId>{"GLOBAL",0}); //GLOBAL scope 0 is automatically open (top scope)
 tensor_rt_->openScope("GLOBAL");
}

NumServer::~NumServer()
{
 tensor_rt_->closeScope();
 scopes_.pop();
}

void NumServer::reconfigureTensorRuntime(const std::string & dag_executor_name,
                                         const std::string & node_executor_name)
{
 tensor_rt_ = std::move(std::make_shared<runtime::TensorRuntime>(dag_executor_name,node_executor_name));
 return;
}

void NumServer::registerTensorMethod(std::shared_ptr<TensorMethod> method)
{
 auto res = ext_methods_.insert({method->name(),method});
 if(!(std::get<1>(res))) std::cout << "#ERROR(NumServer::registerTensorMethod): Method already exists: " <<
  method->name() << std::endl;
 assert(std::get<1>(res));
 return;
}

std::shared_ptr<TensorMethod> NumServer::getTensorMethod(const std::string & tag)
{
 return ext_methods_[tag];
}

void NumServer::registerExternalData(const std::string & tag, std::shared_ptr<BytePacket> packet)
{
 auto res = ext_data_.insert({tag,packet});
 if(!(std::get<1>(res))) std::cout << "#ERROR(NumServer::registerExternalData): Data already exists: " << tag << std::endl;
 assert(std::get<1>(res));
 return;
}

std::shared_ptr<BytePacket> NumServer::getExternalData(const std::string & tag)
{
 return ext_data_[tag];
}


ScopeId NumServer::openScope(const std::string & scope_name)
{
 assert(scope_name.length() > 0);
 ScopeId new_scope_id = scopes_.size();
 scopes_.push(std::pair<std::string,ScopeId>{scope_name,new_scope_id});
 return new_scope_id;
}

ScopeId NumServer::closeScope()
{
 assert(!scopes_.empty());
 const auto & prev_scope = scopes_.top();
 ScopeId prev_scope_id = std::get<1>(prev_scope);
 scopes_.pop();
 return prev_scope_id;
}


SpaceId NumServer::createVectorSpace(const std::string & space_name, DimExtent space_dim,
                                     const VectorSpace ** space_ptr)
{
 assert(space_name.length() > 0);
 SpaceId space_id = space_register_.registerSpace(std::make_shared<VectorSpace>(space_dim,space_name));
 if(space_ptr != nullptr) *space_ptr = space_register_.getSpace(space_id);
 return space_id;
}

void NumServer::destroyVectorSpace(const std::string & space_name)
{
 assert(false);
 //`Finish
 return;
}

void NumServer::destroyVectorSpace(SpaceId space_id)
{
 assert(false);
 //`Finish
 return;
}

const VectorSpace * NumServer::getVectorSpace(const std::string & space_name) const
{
 return space_register_.getSpace(space_name);
}


SubspaceId NumServer::createSubspace(const std::string & subspace_name,
                                     const std::string & space_name,
                                     std::pair<DimOffset,DimOffset> bounds,
                                     const Subspace ** subspace_ptr)
{
 assert(subspace_name.length() > 0 && space_name.length() > 0);
 const VectorSpace * space = space_register_.getSpace(space_name);
 assert(space != nullptr);
 SubspaceId subspace_id = space_register_.registerSubspace(std::make_shared<Subspace>(space,bounds,subspace_name));
 if(subspace_ptr != nullptr) *subspace_ptr = space_register_.getSubspace(space_name,subspace_name);
 auto res = subname2id_.insert({subspace_name,space->getRegisteredId()});
 if(!(res.second)) std::cout << "#ERROR(NumServer::createSubspace): Subspace already exists: " << subspace_name << std::endl;
 assert(res.second);
 return subspace_id;
}

void NumServer::destroySubspace(const std::string & subspace_name)
{
 assert(false);
 //`Finish
 return;
}

void NumServer::destroySubspace(SubspaceId subspace_id)
{
 assert(false);
 //`Finish
 return;
}

const Subspace * NumServer::getSubspace(const std::string & subspace_name) const
{
 assert(subspace_name.length() > 0);
 auto it = subname2id_.find(subspace_name);
 if(it == subname2id_.end()) std::cout << "#ERROR(NumServer::getSubspace): Subspace not found: " << subspace_name << std::endl;
 assert(it != subname2id_.end());
 SpaceId space_id = (*it).second;
 const VectorSpace * space = space_register_.getSpace(space_id);
 assert(space != nullptr);
 const std::string & space_name = space->getName();
 assert(space_name.length() > 0);
 return space_register_.getSubspace(space_name,subspace_name);
}

void NumServer::submit(std::shared_ptr<TensorOperation> operation)
{
 assert(operation);
 if(operation->getOpcode() == TensorOpCode::CREATE){ //TENSOR_CREATE sets tensor element type for future references
  auto tensor = operation->getTensorOperand(0);
  auto elem_type = std::dynamic_pointer_cast<numerics::TensorOpCreate>(operation)->getTensorElementType();
  tensor->setElementType(elem_type);
 }
 tensor_rt_->submit(operation);
 return;
}

void NumServer::submit(TensorNetwork & network)
{
 auto & op_list = network.getOperationList();
 for(auto op = op_list.begin(); op != op_list.end(); ++op){
  tensor_rt_->submit(*op);
 }
 return;
}

void NumServer::submit(std::shared_ptr<TensorNetwork> network)
{
 assert(network);
 return submit(*network);
}

bool NumServer::sync(const Tensor & tensor, bool wait)
{
 return tensor_rt_->sync(tensor,wait);
}

bool NumServer::sync(TensorOperation & operation, bool wait)
{
 return tensor_rt_->sync(operation,wait);
}

bool NumServer::sync(TensorNetwork & network, bool wait)
{
 return sync(*(network.getTensor(0)),wait);
}

bool NumServer::sync(const std::string & name, bool wait)
{
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()) return false;
 return sync(*(iter->second),wait);
}

Tensor & NumServer::getTensorRef(const std::string & name)
{
 auto iter = tensors_.find(name);
 assert(iter != tensors_.end());
 return *(iter->second);
}

TensorElementType NumServer::getTensorElementType(const std::string & name) const
{
 auto iter = tensors_.find(name);
 assert(iter != tensors_.end());
 return (iter->second)->getElementType();
}

bool NumServer::destroyTensor(const std::string & name)
{
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()) return false;
 std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::DESTROY);
 op->setTensorOperand(iter->second);
 submit(op);
 return true;
}

bool NumServer::transformTensor(const std::string & name, std::shared_ptr<TensorMethod> functor)
{
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()) return false;
 std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::TRANSFORM);
 op->setTensorOperand(iter->second);
 std::dynamic_pointer_cast<numerics::TensorOpTransform>(op)->resetFunctor(functor);
 submit(op);
 return true;
}

bool NumServer::evaluateTensorNetwork(const std::string & name, const std::string & network)
{
 std::vector<std::string> tensors;
 auto parsed = parse_tensor_network(network,tensors);
 if(parsed){
  std::map<std::string,std::shared_ptr<Tensor>> tensor_map;
  std::string tensor_name;
  std::vector<IndexLabel> indices;
  for(const auto & tensor: tensors){
   bool complex_conj;
   parsed = parse_tensor(tensor,tensor_name,indices,complex_conj);
   if(!parsed) break;
   auto iter = tensors_.find(tensor_name);
   if(iter == tensors_.end()) parsed = false;
   if(!parsed) break;
   auto res = tensor_map.emplace(std::make_pair(tensor_name,iter->second));
   parsed = res.second; if(!parsed) break;
  }
  if(parsed){
   TensorNetwork tensnet(name,network,tensor_map);
   submit(tensnet);
  }
 }
 return parsed;
}

} //namespace exatn
