#include <gtest/gtest.h>

#include "MPIClient.hpp"
#include "exatn_config.hpp"
#include "MPIServer.hpp"

using namespace exatn;
using namespace exatn::rpc::mpi;

TEST(MPIRPCTester, checkSimple) {

//  MPIClient client;
 int world_size;
 MPI_Comm_size(MPI_COMM_WORLD, &world_size);

 // Get the rank of the process
 int world_rank;
 MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

 std::cout << "SIZES: " << world_size << ", " << world_rank << "\n";

}

int main(int argc, char **argv) {
  MPI_Init(&argc,&argv);

  ::testing::InitGoogleTest(&argc, argv);

  auto ret = RUN_ALL_TESTS();

  MPI_Finalize();
  return ret;
}
