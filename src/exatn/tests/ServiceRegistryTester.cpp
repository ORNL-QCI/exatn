#include <gtest/gtest.h>

#include "ServiceRegistry.hpp"
#include "exatn_config.hpp"
#include "TestInterface.hpp"

using namespace exatn;
using namespace exatn::utility;

TEST(ServiceRegistryTester, checkInitialize) {

 std::string fakepluginpath = std::string(EXATN_BUILD_DIR) + "/src/exatn/tests/testplugin";

 ServiceRegistry registry;
 registry.initialize(fakepluginpath);
 auto test = registry.getService<TestInterface>("test");
 auto s = test->test("HOWDY");
 EXPECT_EQ("HOWDY",s);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
