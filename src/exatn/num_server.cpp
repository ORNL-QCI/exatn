/** ExaTN::Numerics: Numerical server
REVISION: 2019/08/04

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "num_server.hpp"

#include "assert.h"

namespace exatn{

std::shared_ptr<NumServer> numericalServer = std::make_shared<NumServer>();

NumServer::NumServer():
 tensor_rt_(std::make_shared<runtime::TensorRuntime>())
{
 scopes_.push(std::pair<std::string,ScopeId>{"GLOBAL",0}); //GLOBAL scope 0 is automatically open (top scope)
}

void NumServer::reconfigureTensorRuntime(const std::string & dag_executor_name,
                                         const std::string & node_executor_name)
{
 tensor_rt_ = std::move(std::make_shared<runtime::TensorRuntime>(dag_executor_name,node_executor_name));
 return;
}

void NumServer::registerTensorMethod(std::shared_ptr<TensorMethod<Identifiable>> method)
{
 auto res = ext_methods_.insert({method->name(),method});
 if(!(std::get<1>(res))) std::cout << "#ERROR(NumServer::registerTensorMethod): Method already exists: " <<
  method->name() << std::endl;
 assert(std::get<1>(res));
 return;
}

std::shared_ptr<TensorMethod<Identifiable>> NumServer::getTensorMethod(const std::string & tag)
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
 tensor_rt_->submit(operation);
 return;
}

void NumServer::submit(TensorNetwork & network)
{
 assert(false);//`Finish
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
 assert(false);//`Finish
 return false;
}

} //namespace exatn
