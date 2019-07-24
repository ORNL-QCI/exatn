#include "node_executor_talsh.hpp"
#include "node_executor_exatensor.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"

#include <memory>
#include <set>

using namespace cppmicroservices;

namespace {

/**
 */
class US_ABI_LOCAL NodeExecutorActivator : public BundleActivator {

public:
  NodeExecutorActivator() {}

  /**
   */
  void Start(BundleContext context) {

    auto g1 = std::make_shared<exatn::runtime::TalshNodeExecutor>();
    auto g2 = std::make_shared<exatn::runtime::ExatensorNodeExecutor>();

    context.RegisterService<exatn::runtime::TensorNodeExecutor>(g1);
    context.RegisterService<exatn::runtime::TensorNodeExecutor>(g2);
  }

  /**
   */
  void Stop(BundleContext /*context*/) {}
};

} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(NodeExecutorActivator)
