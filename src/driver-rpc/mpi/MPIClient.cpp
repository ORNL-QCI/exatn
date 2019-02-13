#include "MPIClient.hpp"

namespace exatn {
namespace rpc {
namespace mpi {

MPIClient::MPIClient() {
//   std::string portName = "tcp://172.17.0.2:52893";
//   char * c_portName = const_cast<char*> (portName.c_str());
  char portName[MPI_MAX_PORT_NAME];
  MPI_Lookup_name("exatn-mpi-server", MPI_INFO_NULL, portName);
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
  MPI_Send(0, 1, MPI_INT, 0, -1, serverComm);
  MPI_Comm_disconnect(&serverComm);
}

} // namespace mpi
} // namespace rpc
} // namespace exatn