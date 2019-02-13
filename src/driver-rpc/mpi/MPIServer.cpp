#include "MPIServer.hpp"
#include <future>

namespace exatn {
namespace rpc {
namespace mpi {

void MPIServer::start() {

  listen = true;

  MPI_Comm client;
  MPI_Status status;

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  char buf[1000];
  char portName[MPI_MAX_PORT_NAME];

  MPI_Open_port(MPI_INFO_NULL, portName);
  std::cout << size << ", [server] starting server " << portName << "\n";

  MPI_Send(portName, MPI_MAX_PORT_NAME, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
  MPI_Comm_accept(portName, MPI_INFO_NULL, 0, MPI_COMM_SELF, &client);

  while (listen) {
    std::cout << "[server] accepting incoming connection.\n";

    std::cout << "[server] Listening for requests.\n";
    std::cout << "[server] Running MPI_Recv.\n";

    MPI_Recv(buf, 1000, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, client, &status);
    std::cout << "[server] received: " << std::string(buf) << "\n";

    if (status.MPI_TAG == 1) {
      std::cout << "[server] received stop command\n";
      stop();
    } else {

      std::cout << "[server] Execution taprol commands.\n";

      // FIXME with DMITRY:
      // Execute the TAPROL with our Numerics backend
      // I'm assuming the result will be a contracted
      // scalar (double).

      // Now take that result and execute an
      // asynchronous Isend back to the client rank 0

      MPI_Request request;
      std::cout << "[server] processed taprol, returning result.\n";
      double d = 3.3;
      MPI_Isend(&d, 1, MPI_DOUBLE, 0, 0, client,
            &request);
    }
  }

  std::cout << "[server] Out of event loop.\n";
  MPI_Comm_disconnect(&client);
  return;
}

void MPIServer::stop() { listen = false; }

} // namespace mpi
} // namespace rpc
} // namespace exatn