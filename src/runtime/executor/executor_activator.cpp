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

    //Activate tensor graph (DAG) executors:
    context.RegisterService<exatn::runtime::TensorGraphExecutor>(
      std::make_shared<exatn::runtime::EagerGraphExecutor>()
    );
    context.RegisterService<exatn::runtime::TensorGraphExecutor>(
      std::make_shared<exatn::runtime::LazyGraphExecutor>()
    );

    //Activate tensor graph (DAG) node executors:
    context.RegisterService<exatn::runtime::TensorNodeExecutor>(
      std::make_shared<exatn::runtime::TalshNodeExecutor>()
    );
    context.RegisterService<exatn::runtime::TensorNodeExecutor>(
      std::make_shared<exatn::runtime::ExatensorNodeExecutor>()
    );
  }

  void Stop(BundleContext /*context*/) {}
};

} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(ExecutorActivator)
