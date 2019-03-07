#include "exatn.hpp"

namespace exatn {

bool exatnFrameworkInitialized = false;

std::shared_ptr<ServiceRegistry> serviceRegistry =
    std::make_shared<ServiceRegistry>();

void Initialize() {
  if (!exatnFrameworkInitialized) {
    serviceRegistry->initialize();
    exatnFrameworkInitialized =true;
  }
}
bool isInitialize() {
    return exatnFrameworkInitialized;
}

void Finalize() { exatnFrameworkInitialized = false; }

} // namespace exatn