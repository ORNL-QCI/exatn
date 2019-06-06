#include <gtest/gtest.h>

#include "numerics.hpp"

#include <iostream>
#include <utility>

using namespace exatn;
using namespace exatn::numerics;

TEST(NumericsTester, checkSimple)
{
 {
  TensorSignature signa{std::pair<SpaceId,SubspaceId>(1,5),
                        std::pair<SpaceId,SubspaceId>(SOME_SPACE,13)};
  std::cout << signa.getRank() << " " << signa.getDimSpaceId(0) << " "
            << signa.getDimSubspaceId(1) << " "
            << std::get<0>(signa.getDimSpaceAttr(1)) << std::endl;
  signa.printIt();
  std::cout << std::endl;

  TensorShape shape{61,32};
  std::cout << shape.getRank() << " " << shape.getDimExtent(0) << " "
            << shape.getDimExtent(1) << std::endl;
  shape.printIt();
  std::cout << std::endl;

  TensorLeg leg{1,4};
  leg.printIt();
  std::cout << std::endl;
 }
}

TEST(NumericsTester, checkNumServer)
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
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
