#include "MPIClient.hpp"

using namespace exatn::rpc::mpi;

int main(int argc, char** argv) {

  MPI_Init(&argc, &argv);

  MPIClient client;
  auto jobId = client.sendTaProl("hello world");
  std::cout << "[client.cpp] job-id = " << jobId << ".\n";

//   auto value = client.retrieveResult(jobId);

  client.shutdown();

  MPI_Finalize();

}
