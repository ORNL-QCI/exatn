#include "exatn_service.hpp"

namespace exatn {

bool exatnFrameworkInitialized = false;

std::shared_ptr<ServiceRegistry> serviceRegistry = std::make_shared<ServiceRegistry>();

} // namespace exatn
