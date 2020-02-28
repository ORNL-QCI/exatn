#include "TAProLInterpreter.hpp"
#include <gtest/gtest.h>
#include "exatn.hpp"

using namespace exatn::parser;

TEST(TAProLInterpreterTester, checkSimple) {


  TAProLInterpreter interpreter;
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

  interpreter.interpret(src);
}

int main(int argc, char **argv) {
#ifdef MPI_ENABLED
  int mpi_error = MPI_Init(&argc, &argv); assert(mpi_error == MPI_SUCCESS);
#endif
  exatn::initialize();

  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();

  exatn::finalize();
#ifdef MPI_ENABLED
  mpi_error = MPI_Finalize(); assert(mpi_error == MPI_SUCCESS);
#endif
  return ret;
}
