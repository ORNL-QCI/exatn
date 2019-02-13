#include "MPIClient.hpp"
#include <gtest/gtest.h>

using namespace exatn::rpc::mpi;

TEST(client_test, checkSimple) {

// int main(int argc, char** argv) {


  // Create the client
  MPIClient client;

  // Send some taprol asynchronously, here its just
  // a test so just send any string
  auto jobId = client.sendTaProl("hello world");

  std::cout << "[client.cpp] job-id = " << jobId << ".\n";

  // Retrieve the result
  auto value = client.retrieveResult(jobId);
  EXPECT_EQ(3.3, value);

  std::cout << "[client.cpp] value is " << value << "\n";

  // Try it again!
  jobId = client.sendTaProl("hello again");
  value = client.retrieveResult(jobId);

  EXPECT_EQ(3.3, value);
  std::cout << "[client.cpp] value second time is " << value << "\n";

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
