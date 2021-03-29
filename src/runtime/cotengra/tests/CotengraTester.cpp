#include "contraction_seq_optimizer.hpp"
#include "exatn.hpp"
#include <gtest/gtest.h>

TEST(CotengraTester, checkSimple) {
  std::cout << "HOWDY\n";
  auto cotengra =
      exatn::getService<exatn::numerics::ContractionSeqOptimizer>("cotengra");
  exatn::TensorNetwork tenNet;
  std::list<exatn::numerics::ContrTriple> results;
  cotengra->determineContractionSequence(tenNet, results,
                                         []() -> unsigned int { return 1; });
}

int main(int argc, char **argv) {
  exatn::initialize();

  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();

  exatn::finalize();
  return ret;
}
