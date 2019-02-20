#include "MPIClient.hpp"
#include "MPIServer.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"

#include <memory>
#include <set>

using namespace cppmicroservices;

namespace {

/**
 */
class US_ABI_LOCAL MPIRPCActivator : public BundleActivator {

public:
  MPIRPCActivator() {}

  /**
   */
  void Start(BundleContext context) {

    auto s = std::make_shared<exatn::rpc::mpi::MPIServer>();
    context.RegisterService<exatn::rpc::DriverServer>(s);

    auto c = std::make_shared<exatn::rpc::mpi::MPIClient>();
    context.RegisterService<exatn::rpc::DriverClient>(c);
  }

  /**
   */
  void Stop(BundleContext /*context*/) {}
};

} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(MPIRPCActivator)