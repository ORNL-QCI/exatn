/** ExaTN::Numerics: Numerical server
REVISION: 2019/05/27

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

} //namespace numerics

} //namespace exatn
