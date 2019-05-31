/** ExaTN::Numerics: Numerical server
REVISION: 2019/05/31

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "num_server.hpp"

namespace exatn{

namespace numerics{

void NumServer::registerTensorMethod(std::shared_ptr<TensorMethod<Identifiable>> method)
{
 ext_methods_.insert({method->name(),method});
 return;
}

std::shared_ptr<TensorMethod<Identifiable>> NumServer::getTensorMethod(const std::string & tag)
{
 return ext_methods_[tag];
}

void NumServer::registerExternalData(const std::string & tag, std::shared_ptr<BytePacket> packet)
{
 ext_data_.insert({tag,packet});
 return;
}

std::shared_ptr<BytePacket> NumServer::getExternalData(const std::string & tag)
{
 return ext_data_[tag];
}


ScopeId NumServer::openScope(const std::string & scope_name)
{
 //`Finish
 return 0;
}

ScopeId NumServer::closeScope()
{
 auto prev_scope = scopes_.top();
 scopes_.pop();
 return prev_scope;
}


SpaceId NumServer::createVectorSpace(const std::string & space_name, DimExtent space_dim,
                                     const VectorSpace ** space_ptr)
{
 //`Finish
 return 0;
}

void NumServer::destroyVectorSpace(const std::string & space_name)
{
 //`Finish
 return;
}

void NumServer::destroyVectorSpace(SpaceId space_id)
{
 //`Finish
 return;
}


SubspaceId NumServer::createSubspace(const std::string & subspace_name,
                                     const std::string & space_name,
                                     const std::pair<DimOffset,DimOffset> bounds,
                                     const Subspace ** subspace_ptr)
{
 //`Finish
 return 0;
}

void NumServer::destroySubspace(const std::string & subspace_name)
{
 //`Finish
 return;
}

void NumServer::destroySubspace(SubspaceId subspace_id)
{
 //`Finish
 return;
}

} //namespace numerics

} //namespace exatn
