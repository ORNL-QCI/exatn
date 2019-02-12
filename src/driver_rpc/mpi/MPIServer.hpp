#ifndef EXATN_DRIVER_MPISERVER_HPP_
#define EXATN_DRIVER_MPISERVER_HPP_

#include "DriverServer.hpp"
#include "mpi.h"
#include <string>

namespace exatn {
namespace rpc {
namespace mpi {
class MPIServer : public DriverServer {

protected:
  bool listen = false;

  MPI_Comm communicator;
  std::string portName = "exatn-mpi-driver";

public:
  MPIServer() {}

  void setCommunicator(MPI_Comm &comm) { communicator = comm; }

  void start() override;
  void stop() override;

  const std::string name() const override { return "mpi"; }
  const std::string description() const override { return ""; }
};
} // namespace mpi
} // namespace rpc
} // namespace exatn
#endif