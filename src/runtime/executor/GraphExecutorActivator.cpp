#include "graph_executor_eager.hpp"
#include "graph_executor_lazy.hpp"

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

    auto g1 = std::make_shared<exatn::runtime::EagerGraphExecutor>();
    auto g2 = std::make_shared<exatn::runtime::LazyGraphExecutor>();

    context.RegisterService<exatn::runtime::TensorGraphExecutor>(g1);
    context.RegisterService<exatn::runtime::TensorGraphExecutor>(g2);
  }

  /**
   */
  void Stop(BundleContext /*context*/) {}
};

} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(GraphExecutorActivator)
