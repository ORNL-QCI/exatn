#include "graph_executor_eager.hpp"
#include "graph_executor_lazy.hpp"
#include "node_executor_exatensor.hpp"
#include "node_executor_talsh.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"

#include <memory>
#include <set>

using namespace cppmicroservices;

namespace {

/**
 */
class US_ABI_LOCAL ExecutorActivator : public BundleActivator {

public:
  ExecutorActivator() {}

  /**
   */
  void Start(BundleContext context) {

    auto gex1 = std::make_shared<exatn::runtime::EagerGraphExecutor>();
    auto gex2 = std::make_shared<exatn::runtime::LazyGraphExecutor>();

    context.RegisterService<exatn::runtime::TensorGraphExecutor>(gex1);
    context.RegisterService<exatn::runtime::TensorGraphExecutor>(gex2);

    auto nex1 = std::make_shared<exatn::runtime::TalshNodeExecutor>();
    auto nex2 = std::make_shared<exatn::runtime::ExatensorNodeExecutor>();

    context.RegisterService<exatn::runtime::TensorNodeExecutor>(nex1);
    context.RegisterService<exatn::runtime::TensorNodeExecutor>(nex2);
  }

  /**
   */
  void Stop(BundleContext /*context*/) {}
};

} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(ExecutorActivator)
