#include "exatn_service.hpp"

namespace exatn {

bool exatnFrameworkInitialized = false;

std::shared_ptr<ServiceRegistry> serviceRegistry = std::make_shared<ServiceRegistry>();


void initialize() {
  if(!exatnFrameworkInitialized) {
    serviceRegistry->initialize();
    exatnFrameworkInitialized = true;
  }
}


bool isInitialized() {
  return exatnFrameworkInitialized;
}


void finalize() {
  exatnFrameworkInitialized = false;
}

} // namespace exatn
