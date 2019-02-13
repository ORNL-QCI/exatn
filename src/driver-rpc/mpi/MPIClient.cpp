#include "MPIClient.hpp"

namespace exatn {
namespace rpc {
namespace mpi {

MPIClient::MPIClient() {
  char portName[MPI_MAX_PORT_NAME];

  MPI_Status status;

  MPI_Recv(portName, MPI_MAX_PORT_NAME, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
  std::cout << "[client] Attempting to connect with server - " << portName << "\n";

  MPI_Comm_connect(portName, MPI_INFO_NULL, 0, MPI_COMM_SELF,
                   &serverComm);
  std::cout << "[client] Connected with the server\n";
}

// Send TaProl string, get a jobId string,
// so this is an asynchronous call
const std::string MPIClient::sendTaProl(const std::string taProlStr) {

  MPI_Request request;
  auto jobId = generateRandomString();
  std::cout << "[client] sending request with jobid " << jobId << "\n";
  MPI_Isend(taProlStr.c_str(), taProlStr.size(), MPI_CHAR, 0, 0, serverComm,
            &request);
  requests.insert({jobId, request});
  return jobId;
}

// Retrieve result of job with given jobId.
// Returns a scalar type double?
const double MPIClient::retrieveResult(const std::string jobId) {

  auto request = requests[jobId];

  MPI_Status status;
  MPI_Wait(&request, &status);

  // now we know the execution has occurred,
  // so get the result with a Recv.
}

void MPIClient::shutdown() {
  char buf[1];
  MPI_Request request;

  std::cout << "[client] sending shutdown.\n";
  MPI_Isend(buf, 1, MPI_CHAR, 0, 1, serverComm, &request);
   MPI_Status status;
   std::cout << "[client] waiting for shutdown.\n";
  MPI_Wait(&request, &status);
  MPI_Comm_disconnect(&serverComm);
}

} // namespace mpi
} // namespace rpc
} // namespace exatn