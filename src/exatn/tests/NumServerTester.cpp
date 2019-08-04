#include <gtest/gtest.h>

#include "exatn.hpp"

#include <iostream>
#include <utility>

using namespace exatn;
using namespace exatn::numerics;

TEST(NumServerTester, checkNumServer)
{

 const VectorSpace * space1;
 auto space1_id = numericalServer->createVectorSpace("Space1",1024,&space1);
 space1->printIt();
 std::cout << std::endl;

 const VectorSpace * space2;
 auto space2_id = numericalServer->createVectorSpace("Space2",2048,&space2);
 space2->printIt();
 std::cout << std::endl;

 const Subspace * subspace1;
 auto subspace1_id = numericalServer->createSubspace("S11","Space1",{13,246},&subspace1);
 subspace1->printIt();
 std::cout << std::endl;

 const Subspace * subspace2;
 auto subspace2_id = numericalServer->createSubspace("S21","Space2",{1056,1068},&subspace2);
 subspace2->printIt();
 std::cout << std::endl;

 const VectorSpace * space = numericalServer->getVectorSpace("");
 space->printIt();
 std::cout << std::endl;

 space = numericalServer->getVectorSpace("Space2");
 space->printIt();
 std::cout << std::endl;

 const Subspace * subspace = numericalServer->getSubspace("S11");
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
