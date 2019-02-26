#include "DriverClient.hpp"
#include <gtest/gtest.h>
#include "exatn.hpp"
#include "mpi.h"

using namespace exatn::rpc;

const std::string src = R"src(entry: main
scope main group()
 subspace(): s0=[0:127]
 index(s0): a,b,c,d,i,j,k,l
 H2(a,b,c,d) = method("HamiltonianTest")
 T2(a,b,c,d) = {1.0,0.0}
 Z2(a,b,c,d) = {0.0,0.0}
 Z2(a,b,c,d) += H2(i,j,k,l) * T2(c,d,i,j) * T2(a,b,k,l)
 X2() = {0.0,0.0}
 X2() += Z2+(a,b,c,d) * Z2(a,b,c,d)
 save X2: tag("Z2_norm")
 ~X2
 ~Z2
 ~T2
 ~H2
end scope main)src";

TEST(client_test, checkSimple) {

  exatn::Initialize();

  auto tm = exatn::getService<TensorMethod<exatn::Identifiable>>("HamiltonianTest");

  // How to set the state of the TensorMethod...
//   BytePacket p;
//   initBytePacket(&p);
//   appendToBytePacket(&p, .002);
//   tm->unpack(p);

  // Create the client
  auto client = exatn::getService<DriverClient>("mpi");
  client->registerTensorMethod(*tm.get());

  // Send some taprol asynchronously, here its just
  // a test so just send any string
  auto jobId = client->interpretTAProL(src);

  std::cout << "[client.cpp] job-id = " << jobId << ".\n";

  // Retrieve the result
  auto values = client->getResults(jobId);
  EXPECT_EQ(3.3, std::real(values[0]));
  EXPECT_EQ(3.3, std::imag(values[0]));

  std::cout << "[client.cpp] value is " << std::real(values[0]) << ", " << std::imag(values[0]) << "\n";

  // Try it again!
  jobId = client->interpretTAProL(src);
  values = client->getResults(jobId);

  EXPECT_EQ(3.3, std::real(values[0]));
  EXPECT_EQ(3.3, std::imag(values[0]));
  std::cout << "[client.cpp] value second time is " << std::real(values[0]) << ", " << std::imag(values[0]) << "\n";

  // Shutdown the client, this
  // also tells the server to shutdown.
  client->shutdown();

  exatn::Finalize();

}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  MPI_Finalize();
  return ret;
}
