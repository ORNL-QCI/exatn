/** ExaTN::Numerics: Tensor network builder factory
REVISION: 2019/07/11

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "network_build_factory.hpp"

namespace exatn{

namespace numerics{

NetworkBuildFactory::NetworkBuildFactory()
{
 registerNetworkBuilder("MPS",&NetworkBuilderMPS::createNew);
}

void NetworkBuildFactory::registerNetworkBuilder(const std::string & name, createNetworkBuilderFn creator)
{
 factory_map_[name] = creator;
 return;
}

std::unique_ptr<NetworkBuilder> NetworkBuildFactory::createNetworkBuilder(const std::string & name)
{
 auto it = factory_map_.find(name);
 if(it != factory_map_.end()) return (it->second)();
 return std::unique_ptr<NetworkBuilder>(nullptr);
}

NetworkBuildFactory * NetworkBuildFactory::get()
{
 static NetworkBuildFactory single_instance;
 return &single_instance;
}

} //namespace numerics

} //namespace exatn
