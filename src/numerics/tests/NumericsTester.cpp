#include <gtest/gtest.h>

#include "tensor_factory.hpp"
#include "tensor.hpp"
#include "tensor_signature.hpp"
#include "tensor_shape.hpp"
#include "tensor_leg.hpp"

#include <iostream>
#include <utility>

using namespace exatn;
using namespace exatn::numerics;

TEST(NumericsTester, checkSimple) {

 TensorSignature signa{std::pair<SpaceId,SubspaceId>(1,5), std::pair<SpaceId,SubspaceId>(SOME_SPACE,13)};
 std::cout << signa.getRank() << " " <<
              signa.getDimSpaceId(0) << " " <<
              signa.getDimSubspaceId(1) << " " <<
              std::get<0>(signa.getDimSpaceAttr(1)) << std::endl;
 signa.printIt(); std::cout << std::endl;

 TensorShape shape{61,32};
 std::cout << shape.getRank() << " " <<
              shape.getDimExtent(0) << " " << shape.getDimExtent(1) << std::endl;
 shape.printIt(); std::cout << std::endl;

 TensorLeg leg{1,4};
 leg.printIt(); std::cout << std::endl;

 TensorFactory tensor_factory;

 auto tensor = tensor_factory.createTensor(TensorKind::TENSOR,"T",shape,signa);
 tensor->printIt(); std::cout << std::endl;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
