#include "talsh_backend.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"

#include <memory>
#include <set>

using namespace cppmicroservices;

namespace {

/**
 */
class US_ABI_LOCAL TalshActivator : public BundleActivator {

public:
  TalshActivator() {}

  /**
   */
  void Start(BundleContext context) {
    auto t = std::make_shared<exatn::numerics::talsh::TalshBackend>();
    context.RegisterService<exatn::numerics::Backend>(t);
  }

  /**
   */
  void Stop(BundleContext /*context*/) {}
};

} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(TalshActivator)