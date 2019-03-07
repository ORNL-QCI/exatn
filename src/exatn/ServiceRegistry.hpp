#ifndef EXATN_SERVICEREGISTRY_HPP_
#define EXATN_SERVICEREGISTRY_HPP_

#include "Identifiable.hpp"

#include <cppmicroservices/Bundle.h>
#include <cppmicroservices/BundleContext.h>
#include <cppmicroservices/BundleImport.h>
#include <cppmicroservices/Framework.h>
#include <cppmicroservices/FrameworkFactory.h>

using namespace cppmicroservices;

namespace exatn {

class ServiceRegistry {

protected:
  Framework framework;
  BundleContext context;
  std::map<std::string, std::string> installed;
  bool initialized = false;

public:
  ServiceRegistry() : framework(FrameworkFactory().NewFramework()) {}

  void initialize(const std::string pluginPath = "");

  template <typename ServiceInterface> bool hasService(const std::string name) {
    auto allServiceRefs = context.GetServiceReferences<ServiceInterface>();
    for (auto s : allServiceRefs) {
      auto service = context.GetService(s);
      auto identifiable =
          std::dynamic_pointer_cast<exatn::Identifiable>(service);
      if (identifiable && identifiable->name() == name) {
        return true;
      }
    }
    return false;
  }

  template <typename ServiceInterface>
  std::shared_ptr<ServiceInterface> getService(const std::string name) {
    std::shared_ptr<ServiceInterface> ret;
    auto allServiceRefs = context.GetServiceReferences<ServiceInterface>();
    for (auto s : allServiceRefs) {
      auto service = context.GetService(s);
      auto identifiable =
          std::dynamic_pointer_cast<exatn::Identifiable>(service);
      if (identifiable && identifiable->name() == name) {
        ret = service;
      }
    }

    if (!ret) {
      std::cerr << "Could not find service with name " << name
                << ". Perhaps the service is not Identifiable.\n";
    }

    return ret;
  }
};

} // namespace exatn

#endif
