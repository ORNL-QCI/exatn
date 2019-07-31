#include <gtest/gtest.h>

#include "exatn.hpp"
#include "numerics.hpp"

#include <iostream>
#include <utility>

using namespace exatn;
using namespace exatn::numerics;

TEST(NumServerTester, checkNumServer)
{

 NumServer num_server;

 const VectorSpace * space1;
 auto space1_id = num_server.createVectorSpace("Space1",1024,&space1);
 space1->printIt();
 std::cout << std::endl;

 const VectorSpace * space2;
 auto space2_id = num_server.createVectorSpace("Space2",2048,&space2);
 space2->printIt();
 std::cout << std::endl;

 const Subspace * subspace1;
 auto subspace1_id = num_server.createSubspace("S11","Space1",{13,246},&subspace1);
 subspace1->printIt();
 std::cout << std::endl;

 const Subspace * subspace2;
 auto subspace2_id = num_server.createSubspace("S21","Space2",{1056,1068},&subspace2);
 subspace2->printIt();
 std::cout << std::endl;

 const VectorSpace * space = num_server.getVectorSpace("");
 space->printIt();
 std::cout << std::endl;

 space = num_server.getVectorSpace("Space2");
 space->printIt();
 std::cout << std::endl;

 const Subspace * subspace = num_server.getSubspace("S11");
 subspace->printIt();
 std::cout << std::endl;

}

int main(int argc, char **argv) {
  exatn::initialize();

  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  exatn::finalize();
  return ret;
}
