/** ExaTN::Numerics: Tensor network builder factory
REVISION: 2019/09/10

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Creates tensor network builders of desired kind.
**/

#ifndef EXATN_NUMERICS_NETWORK_BUILD_FACTORY_HPP_
#define EXATN_NUMERICS_NETWORK_BUILD_FACTORY_HPP_

#include "tensor_basic.hpp"
#include "network_builder.hpp"
#include "network_builder_mps.hpp"

#include <string>
#include <memory>
#include <map>

namespace exatn{

namespace numerics{

class NetworkBuildFactory{
public:

 NetworkBuildFactory(const NetworkBuildFactory &) = delete;
 NetworkBuildFactory & operator=(const NetworkBuildFactory &) = delete;
 NetworkBuildFactory(NetworkBuildFactory &&) noexcept = default;
 NetworkBuildFactory & operator=(NetworkBuildFactory &&) noexcept = default;
 ~NetworkBuildFactory() = default;

 /** Registers a new tensor network builder subtype to produce instances of. **/
 void registerNetworkBuilder(const std::string & name, createNetworkBuilderFn creator);

 /** Creates a new instance of a desired subtype. **/
 std::unique_ptr<NetworkBuilder> createNetworkBuilder(const std::string & name);
 /** Creates a new instance of a desired subtype. **/
 std::shared_ptr<NetworkBuilder> createNetworkBuilderShared(const std::string & name);

 /** Returns a pointer to the NetworkBuildFactory singleton. **/
 static NetworkBuildFactory * get();

private:

 NetworkBuildFactory(); //private ctor

 std::map<std::string,createNetworkBuilderFn> factory_map_;

};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_NETWORK_BUILD_FACTORY_HPP_
