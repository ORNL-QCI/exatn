#ifndef EXATN_SERVICE_HPP_
#define EXATN_SERVICE_HPP_

#include "ServiceRegistry.hpp"

#include <iostream>
#include <memory>

#include "errors.hpp"

namespace exatn {

extern bool exatnFrameworkInitialized;
extern bool exatnInitializedMPI;
extern std::shared_ptr<ServiceRegistry> serviceRegistry;

template <typename Service>
std::shared_ptr<Service> getService(const std::string &serviceName) {
  if(!exatn::exatnFrameworkInitialized) {
    std::cerr << "#FATAL(exatn::service): Unable to get service " << serviceName << std::endl
              << "ExaTN is not initialized: Please execute "
                 "exatn::initialize() before using its API.\n";
    assert(false);
  }
  auto service = serviceRegistry->getService<Service>(serviceName);
  if(!service) {
    std::cerr << "#ERROR(exatn::service): Invalid ExaTN service: " << serviceName
              << " in the Service Registry.\n";
    assert(false);
  }
  return service;
}

template <typename Service>
bool hasService(const std::string &serviceName) {
  if(!exatn::exatnFrameworkInitialized) {
    std::cerr << "#FATAL(exatn::service): Unable to check service "+serviceName << std::endl <<
                 "ExaTN is not initialized: Please execute "
                 "exatn::initialize() before using its API.\n";
    assert(false);
  }
  return serviceRegistry->hasService<Service>(serviceName);
}

} // namespace exatn

#endif //EXATN_SERVICE_HPP_
