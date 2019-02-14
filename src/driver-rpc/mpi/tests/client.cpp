#include "MPIClient.hpp"
#include <gtest/gtest.h>

using namespace exatn::rpc::mpi;

TEST(client_test, checkSimple) {

  // Create the client
  MPIClient client;

  // Send some taprol asynchronously, here its just
  // a test so just send any string
  auto jobId = client.sendTAProL("hello world");

  std::cout << "[client.cpp] job-id = " << jobId << ".\n";

  // Retrieve the result
  auto values = client.retrieveResult(jobId);
  EXPECT_EQ(3.3, std::real(values[0]));
  EXPECT_EQ(3.3, std::imag(values[0]));

  std::cout << "[client.cpp] value is " << std::real(values[0]) << ", " << std::imag(values[0]) << "\n";

  // Try it again!
  jobId = client.sendTAProL("hello again");
  values = client.retrieveResult(jobId);

  EXPECT_EQ(3.3, std::real(values[0]));
  EXPECT_EQ(3.3, std::imag(values[0]));
  std::cout << "[client.cpp] value second time is " << std::real(values[0]) << ", " << std::imag(values[0]) << "\n";

  // Shutdown the client, this
  // also tells the server to shutdown.
  client.shutdown();

}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  MPI_Finalize();
  return ret;
}
