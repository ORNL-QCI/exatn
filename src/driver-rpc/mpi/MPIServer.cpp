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
//   std::string portName = "exatn-mpi-server";
//   char * c_portName = const_cast<char*> (portName.c_str());
  char portName[MPI_MAX_PORT_NAME];

  MPI_Info info;

  MPI_Info_create(&info);
  MPI_Info_set(info, "ompi_global_scope", "true");

  MPI_Open_port(MPI_INFO_NULL, portName);
  MPI_Publish_name("exatn-mpi-server", info, portName);

  std::cout << size << ", [server] starting server " << portName << "\n";
  MPI_Comm_accept(portName, MPI_INFO_NULL, 0, MPI_COMM_SELF, &client);

  while (listen) {
    std::cout << "[server] Listening for requests.\n";
    std::cout << "[server] Running MPI_Recv.\n";

    MPI_Recv(buf, 1000, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, client, &status);
    std::cout << "[server] received: " << std::string(buf) << "\n";

    if (status.MPI_TAG == -1) {
      stop();
    } else {

      std::cout << "Hello World, we were sent the following:\n"
                << std::string(buf) << "\n";

      // Execute an Isend back to the client rank 0
      // containing the results

    }
  }

  MPI_Comm_disconnect(&client);
  return;
}

void MPIServer::stop() { listen = false; }

} // namespace mpi
} // namespace rpc
} // namespace exatn