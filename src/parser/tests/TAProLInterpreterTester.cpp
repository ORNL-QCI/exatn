#include "TAProLInterpreter.hpp"
#include <gtest/gtest.h>
#include "exatn.hpp"

using namespace exatn::parser;

TEST(TAProLInterpreterTester, checkSimple) {

  TAProLInterpreter interpreter;

  const std::string src = R"src(
  entry: main
  scope main group()
   space(complex): my_space=[0:255], your_space=[0:511]
   subspace(my_space): s0=[0:127], s1=[128:255]
   subspace(your_space): r0=[42:84], r1=[484:511]
   index(s0): i,j,k,l
   index(r0): a,b,c,d
   H2(a,i,b,j) = {0.0,0.0}
   H2(a,i,b,j) = method("ComputeTwoBodyHamiltonian")
   T2(a,b,i,j) = std_vector
   Z2(a,b,i,j) = {0.0,0.0}
   Z2(a,b,i,j) += H2(a,k,c,i) * T2(b,c,k,j)
   Z2(a,b,i,j) *= 0.25
   T2(a,b,i,j) += Z2(a,b,i,j)
   talsh_t2 = T2
   X2() = {0.0,0.0}
   X2() += Z2+(a,b,i,j) * Z2(a,b,i,j)
   norm_x2 = norm1(X2)
   talsh_x2 = X2
   save X2: tag("Z2_norm2")
   ~X2
   ~Z2
   destroy T2,H2
  end scope main
  )src";

  interpreter.interpret(src);
}

int main(int argc, char **argv) {
  exatn::initialize();

  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();

  exatn::finalize();
  return ret;
}
