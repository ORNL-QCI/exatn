/** ExaTN::Numerics: Tensor network builder factory
REVISION: 2021/06/25

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "network_build_factory.hpp"

namespace exatn{

namespace numerics{

NetworkBuildFactory::NetworkBuildFactory()
{
 registerNetworkBuilder("MPS",&NetworkBuilderMPS::createNew);
 registerNetworkBuilder("TTN",&NetworkBuilderTTN::createNew);
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

std::shared_ptr<NetworkBuilder> NetworkBuildFactory::createNetworkBuilderShared(const std::string & name)
{
 return std::move(createNetworkBuilder(name));
}

NetworkBuildFactory * NetworkBuildFactory::get()
{
 static NetworkBuildFactory single_instance;
 return &single_instance;
}

} //namespace numerics

} //namespace exatn
