#include "MPIClient.hpp"

using namespace exatn::rpc::mpi;

int main(int argc, char** argv) {

  MPI_Init(&argc, &argv);

  MPIClient client;
  auto jobId = client.sendTaProl("hello world");
  std::cout << "[client.cpp] job-id = " << jobId << ".\n";

  auto value = client.retrieveResult(jobId);

  std::cout << "[client.cpp] value is " << value << "\n";

  jobId = client.sendTaProl("hello again");
  value = client.retrieveResult(jobId);
  std::cout << "[client.cpp] value second time is " << value << "\n";

  client.shutdown();

  MPI_Finalize();

}
