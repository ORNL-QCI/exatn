#include "exatensor_backend.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"

#include <memory>
#include <set>

using namespace cppmicroservices;

namespace {

/**
 */
class US_ABI_LOCAL ExatensorActivator : public BundleActivator {

public:
  ExatensorActivator() {}

  /**
   */
  void Start(BundleContext context) {
    auto t = std::make_shared<exatn::numerics::exatensor::ExatensorBackend>();
    context.RegisterService<exatn::numerics::Backend>(t);
  }

  /**
   */
  void Stop(BundleContext /*context*/) {}
};

} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(ExatensorActivator)