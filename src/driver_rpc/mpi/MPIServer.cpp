#include "MPIServer.hpp"
#include <future>

namespace exatn {
namespace rpc {
namespace mpi {

void MPIServer::start() {

  listen = true;

  MPI_Comm client;
  MPI_Status status;

  char buf[1000];
  char *port_name = "exatn-mpi-server\0";

  MPI_Open_port(MPI_INFO_NULL, port_name);
  MPI_Comm_accept(portName.c_str(), MPI_INFO_NULL, 0, MPI_COMM_WORLD, &client);
  std::async([&]() {
    while (listen) {

      MPI_Recv(buf, 1000, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, client,
               &status);
      if (status.MPI_TAG == -1) {
        stop();
      } else {

        std::cout << "Hello World, we were sent the following:\n"
                  << std::string(buf) << "\n";
      }
    }
  });
  return;
}

void MPIServer::stop() { listen = false; }

} // namespace mpi
} // namespace rpc
} // namespace exatn