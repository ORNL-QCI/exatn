#include <gtest/gtest.h>

#include "ServiceRegistry.hpp"

using namespace exatn;

TEST(ServiceRegistryTester, checkInitialize) {

 ServiceRegistry registry;
 registry.initialize();

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
