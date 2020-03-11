#include "exatn_service.hpp"

namespace exatn {

bool exatnFrameworkInitialized = false;
bool exatnInitializedMPI = false;

std::shared_ptr<ServiceRegistry> serviceRegistry = std::make_shared<ServiceRegistry>();

} // namespace exatn
