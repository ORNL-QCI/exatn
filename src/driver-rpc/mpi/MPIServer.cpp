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
  std::cout << "[mpi-server] starting server at port name " << portName << "\n";

  MPI_Send(portName, MPI_MAX_PORT_NAME, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
  MPI_Comm_accept(portName, MPI_INFO_NULL, 0, MPI_COMM_SELF, &client);

  while (listen) {
    std::cout << "[mpi-server] accepting incoming connection.\n";

    std::cout << "[mpi-server] Listening for requests.\n";
    std::cout << "[mpi-server] Running MPI_Recv.\n";

    MPI_Recv(buf, 1000, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, client, &status);
    std::cout << "[mpi-server] received: " << std::string(buf) << "\n";

    if (status.MPI_TAG == 1) {

        // ExaTensor SYNCHRONIZE commands
       std::cout << "[mpi-server] synchronizing! (to be implemented)\n";


    } else if (status.MPI_TAG == 2) {

      std::cout << "[mpi-server] received stop command\n";
      stop();

    } else {

      std::cout << "[mpi-server] Execution taprol commands.\n";

      // FIXME with DMITRY:
      // Execute the TAPROL with our Numerics backend
      // I'm assuming the result will be a contracted
      // scalar (double).
    //   auto simpleTaProlList = exatn::numerics::translate(taprol_str);

      // depending on backend talsh or exatensor
    //   auto backend = getService<Backend>("talsh");
    //   backend->execute(simpleTaProlList);

      // Now take that result and execute an
      // asynchronous Isend back to the client rank 0

      MPI_Request request;
      std::cout << "[mpi-server] processed taprol, returning result.\n";

      // We have a set of results for each GET in the
      // TAPROL program, so now lets send them
      // to the client as (real,imag) complex numbers
      // FIXME assuming 1 value right now
      double real = 3.3;
      double imag = 3.3;

      MPI_Isend(&real, 1, MPI_DOUBLE, 0, 0, client,
            &request);
      MPI_Isend(&imag, 1, MPI_DOUBLE, 0, 0, client,
            &request);
    }
  }

  std::cout << "[mpi-server] Out of event loop.\n";
  MPI_Comm_disconnect(&client);
  return;
}

void MPIServer::stop() { listen = false; }

} // namespace mpi
} // namespace rpc
} // namespace exatn