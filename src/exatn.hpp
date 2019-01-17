#ifndef EXATN_HPP_
#define EXATN_HPP_

#include <iostream>
#include <memory>

#include "ServiceRegistry.hpp"

namespace exatn {

extern bool exatnFrameworkInitialized;
extern std::shared_ptr<ServiceRegistry> serviceRegistry;

void Initialize();

template <typename Service>
std::shared_ptr<Service> getService(const std::string &serviceName) {
  if (!exatn::exatnFrameworkInitialized) {
    std::cerr << "Exatn not initialized before use. Please execute "
                 "exatn::Initialize() before using API.\n";
  }
  auto service = serviceRegistry->getService<Service>(serviceName);
  if (!service) {
    std::cerr << "Invalid exatn Service. Could not find " << serviceName
              << " in Service Registry.\n";
  }
  return service;
}

template <typename Service> bool hasService(const std::string &serviceName) {
  if (!exatn::exatnFrameworkInitialized) {
    std::cerr << "ExaTN not initialized before use. Please execute "
                 "exatn::Initialize() before using API.\n";
  }
  return serviceRegistry->hasService<Service>(serviceName);
}

void Finalize();

} // namespace exatn
#endif
