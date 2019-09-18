#include "ExampleTensorMethod.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"

#include <memory>
#include <set>

using namespace cppmicroservices;

namespace {

/**
 */
class US_ABI_LOCAL ExampleTensorMethodActivator : public BundleActivator {

public:

  ExampleTensorMethodActivator() {}

  /**
   */
  void Start(BundleContext context) {
    auto t =
        std::make_shared<exatn::test::ExampleTensorMethod>();
    context.RegisterService<talsh::TensorFunctor<exatn::Identifiable>>(t);
  }

  /**
   */
  void Stop(BundleContext /*context*/) {}
};

} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(ExampleTensorMethodActivator)
