#include "MPIServer.hpp"
#include <future>
#include "exatn.hpp"

namespace exatn {
namespace rpc {
namespace mpi {

int MPIServer::SENDTAPROL_TAG = 0;
int MPIServer::REGISTER_TENSORMETHOD = 1;
int MPIServer::SYNC_TAG = 2;
int MPIServer::SHUTDOWN_TAG = 3;

void MPIServer::start() {

  // FIXME Provide hook for specifying this at runtime
  backend = exatn::getService<exatn::numerics::Backend>("talsh");

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

    if (status.MPI_TAG == SYNC_TAG) {

      // ExaTensor SYNCHRONIZE commands
      std::cout << "[mpi-server] synchronizing!\n";

      // FIXME SYNCHRONIZE TALSH/EXATENSOR

      // Now take the results and execute an
      // asynchronous Isend back to the client rank 0
      for (int i = 0; i < nResults; i++) {
        MPI_Request request;
        std::cout << "[mpi-server] processed taprol, returning result.\n";

        // We have a set of results for each GET in the
        // TAPROL program, so now lets send them
        // to the client as (real,imag) complex numbers
        // FIXME assuming 1 value right now
        double real = (i+1)*3.3;
        double imag = (i+1)*3.3;

        MPI_Isend(&real, 1, MPI_DOUBLE, 0, 0, client, &request);
        MPI_Isend(&imag, 1, MPI_DOUBLE, 0, 0, client, &request);
      }

      nResults = 0;

    } else if (status.MPI_TAG == SHUTDOWN_TAG) {

      std::cout << "[mpi-server] received stop command\n";
      stop();

    } else if (status.MPI_TAG == SENDTAPROL_TAG) {

      std::cout << "[mpi-server] Executing taprol commands.\n";

      std::string taProlProg(buf);
      auto position = taProlProg.find("save", 0);
      while (position != std::string::npos) {
        nResults++;
        position = taProlProg.find("save", position + 1);
      }


      std::cout << "[mpi-server] Found " << nResults << " save calls.\n";

      // FIXME with DMITRY:
      // Execute the TAPROL with our Numerics backend
      auto simpleTaProlList = backend->translate(taProlProg);

      // depending on backend talsh or exatensor
      backend->execute(simpleTaProlList);


    } else if (status.MPI_TAG == REGISTER_TENSORMETHOD) {

        std::string tmName(buf);

        std::cout << "[mpi-server] Registering tensor method " << tmName << ".\n";

        MPI_Status status;
        char base_addr[1000];
        int size_bytes;
        MPI_Recv(base_addr, size_bytes, MPI_CHAR, 0, MPI_ANY_TAG, client, &status);

        // Here we assume that ExaTN has been initialized by server.main()
        auto tensor_method = exatn::getService<TensorMethod<Identifiable>>(tmName);

        BytePacket packet;
        packet.base_addr = base_addr;
        packet.size_bytes = size_bytes;
        tensor_method->unpack(packet);

        backend->addTensorMethod(tensor_method);

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