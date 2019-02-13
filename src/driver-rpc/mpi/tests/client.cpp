#include "MPIClient.hpp"

using namespace exatn::rpc::mpi;

int main(int argc, char** argv) {

  MPI_Init(&argc, &argv);

  // Create the client
  MPIClient client;

  // Send some taprol asynchronously, here its just
  // a test so just send any string
  auto jobId = client.sendTaProl("hello world");

  std::cout << "[client.cpp] job-id = " << jobId << ".\n";

  // Retrieve the result
  auto value = client.retrieveResult(jobId);

  std::cout << "[client.cpp] value is " << value << "\n";

  // Try it again!
  jobId = client.sendTaProl("hello again");
  value = client.retrieveResult(jobId);

  std::cout << "[client.cpp] value second time is " << value << "\n";

  // Shutdown the client, this
  // also tells the server to shutdown.
  client.shutdown();

  MPI_Finalize();

}
