#ifndef EXATN_DRIVER_MPISERVER_HPP_
#define EXATN_DRIVER_MPISERVER_HPP_

#include "DriverServer.hpp"
#include "mpi.h"
#include <string>
#include <memory>

namespace exatn {
namespace rpc {
namespace mpi {
class MPIServer : public DriverServer {

protected:
  bool listen = false;
  int nResults = 0;
  static int SYNC_TAG;
  static int SHUTDOWN_TAG;
  static int SENDTAPROL_TAG;
  static int REGISTER_TENSORMETHOD;

  std::string portName = "exatn-mpi-driver";

//   std::map<std::string, std::shared_ptr<TensorMethod>> registeredTensorMethods;

public:
  MPIServer() {}

  void start() override;
  void stop() override;

  const std::string name() const override { return "mpi"; }
  const std::string description() const override { return ""; }
};
} // namespace mpi
} // namespace rpc
} // namespace exatn
#endif