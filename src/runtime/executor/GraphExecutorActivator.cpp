
#include "TalshExecutor.hpp"
#include "ExatensorExecutor.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"

#include <memory>
#include <set>

using namespace cppmicroservices;

namespace {

/**
 */
class US_ABI_LOCAL GraphExecutorActivator : public BundleActivator {

public:
  GraphExecutorActivator() {}

  /**
   */
  void Start(BundleContext context) {

    auto g = std::make_shared<exatn::runtime::TalshExecutor>();
    auto g2 = std::make_shared<exatn::runtime::ExatensorExecutor>();

    context.RegisterService<exatn::runtime::GraphExecutor>(g);
    context.RegisterService<exatn::runtime::GraphExecutor>(g2);
  }

  /**
   */
  void Stop(BundleContext /*context*/) {}
};

} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(GraphExecutorActivator)