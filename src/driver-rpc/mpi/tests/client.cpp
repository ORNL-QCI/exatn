#include "MPIClient.hpp"

using namespace exatn::rpc::mpi;

int main(int argc, char** argv) {

  std::cout << "Entering client code\n";
  MPI_Init(&argc, &argv);
  std::cout << "Instantiating client code\n";

  MPIClient client;

  std::cout << "Starting to send taprol\n";
  auto jobId = client.sendTaProl("hello world");

  std::cout << "JOB ID: " << jobId << "\n";

  auto value = client.retrieveResult(jobId);

  client.shutdown();

  MPI_Finalize();

}