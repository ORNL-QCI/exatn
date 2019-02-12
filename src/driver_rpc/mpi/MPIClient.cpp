#include "MPIClient.hpp"

namespace exatn {
namespace rpc {
namespace mpi {

MPIClient::MPIClient() {
  MPI_Comm_connect("exatn-mpi-server", MPI_INFO_NULL, 0, MPI_COMM_WORLD,
                   &serverComm);
}

// Send TaProl string, get a jobId string,
// so this is an asynchronous call
const std::string MPIClient::sendTaProl(const std::string taProlStr) {

  MPI_Request request;
  auto jobId = generateRandomString();
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
  // so get the result.
}

void MPIClient::shutdown() {
  MPI_Send(0, 1, MPI_INT, 0, -1, serverComm);
  MPI_Comm_disconnect(&serverComm);
}

} // namespace mpi
} // namespace rpc
} // namespace exatn