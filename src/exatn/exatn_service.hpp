#ifndef EXATN_SERVICE_HPP_
#define EXATN_SERVICE_HPP_

#include <iostream>
#include <memory>

#include "ServiceRegistry.hpp"

namespace exatn {

extern bool exatnFrameworkInitialized;
extern std::shared_ptr<ServiceRegistry> serviceRegistry;

void initialize();
bool isInitialized();
void finalize();

template <typename Service>
std::shared_ptr<Service> getService(const std::string &serviceName) {
  if (!exatn::exatnFrameworkInitialized) {
    std::cerr << "ExaTN is not initialized: Please execute "
                 "exatn::initialize() before using its API.\n";
  }
  auto service = serviceRegistry->getService<Service>(serviceName);
  if (!service) {
    std::cerr << "Invalid ExaTN Service: Could not find " << serviceName
              << " in the Service Registry.\n";
  }
  return service;
}

template <typename Service>
bool hasService(const std::string &serviceName) {
  if (!exatn::exatnFrameworkInitialized) {
    std::cerr << "ExaTN is not initialized: Please execute "
                 "exatn::initialize() before using its API.\n";
  }
  return serviceRegistry->hasService<Service>(serviceName);
}

} // namespace exatn

#endif //EXATN_SERVICE_HPP_
