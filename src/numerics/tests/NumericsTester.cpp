#include <gtest/gtest.h>
#include "exatn.hpp"

#include <iostream>
#include <utility>

#include "errors.hpp"

using namespace exatn;
using namespace exatn::numerics;

TEST(NumericsTester, checkSimple)
{
 {
  TensorSignature signa{{1,5},{SOME_SPACE,13}};
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

  auto tensor = makeSharedTensor("H0",TensorShape{2,2,2,2});
 }
}


TEST(NumericsTester, checkTensorNetwork)
{
 //3-site MPS closure with 2-body Hamiltonian applied to sites 0 and 1:
 //Z0() = T0(a,b) * T1(b,c,d) * T2(d,e) * H0(a,c,f,g) * S0(f,h) * S1(h,g,i) * S2(i,e)
 // 0      1         2           3         4             5         6           7  <-- tensor id
 TensorNetwork network("{0,1} 3-site MPS closure",
                       std::make_shared<Tensor>("Z0"), //scalar tensor
                       {} //closed tensor network //no legs
                      );
 network.placeTensor(1,
                     std::make_shared<Tensor>("T0",TensorShape{2,2}),
                     std::vector<TensorLeg>{{4,0},{2,0}}
                    );
 network.placeTensor(2,
                     std::make_shared<Tensor>("T1",TensorShape{2,2,2}),
                     std::vector<TensorLeg>{{1,1},{4,1},{3,0}}
                    );
 network.placeTensor(3,
                     std::make_shared<Tensor>("T2",TensorShape{2,2}),
                     std::vector<TensorLeg>{{2,2},{7,1}}
                    );
 network.placeTensor(4,
                     std::make_shared<Tensor>("H0",TensorShape{2,2,2,2}),
                     std::vector<TensorLeg>{{1,0},{2,1},{5,0},{6,1}}
                    );
 network.placeTensor(5,
                     std::make_shared<Tensor>("S0",TensorShape{2,2}),
                     std::vector<TensorLeg>{{4,2},{6,0}}
                    );
 network.placeTensor(6,
                     std::make_shared<Tensor>("S1",TensorShape{2,2,2}),
                     std::vector<TensorLeg>{{5,1},{4,3},{7,0}}
                    );
 network.placeTensor(7,
                     std::make_shared<Tensor>("S2",TensorShape{2,2}),
                     std::vector<TensorLeg>{{6,2},{3,1}}
                    );
 network.finalize(true);
 network.printIt();
 //Remove tensor #6 to create the optimization environment for MPS tensor S1:
 network.deleteTensor(6);
 network.printIt();
}


TEST(NumericsTester, checkTensorNetworkSymbolic)
{
 //3-site MPS closure with 2-body Hamiltonian applied to sites 0 and 1:
 //Z0() = T0(a,b) * T1(b,c,d) * T2(d,e) * H0(a,c,f,g) * S0(f,h) * S1(h,g,i) * S2(i,e)
 // 0      1         2           3         4             5         6           7  <-- tensor id
 TensorNetwork network("{0,1} 3-site MPS closure", //tensor network name
                       "Z0() = T0(a,b) * T1(b,c,d) * T2(d,e) * H0(a,c,f,g) * S0(f,h) * S1(h,g,i) * S2(i,e)", //tensor network specification
                       std::map<std::string,std::shared_ptr<Tensor>>{
                        {"Z0",std::make_shared<Tensor>("Z0")},
                        {"T0",std::make_shared<Tensor>("T0",TensorShape{2,2})},
                        {"T1",std::make_shared<Tensor>("T1",TensorShape{2,2,2})},
                        {"T2",std::make_shared<Tensor>("T2",TensorShape{2,2})},
                        {"H0",std::make_shared<Tensor>("H0",TensorShape{2,2,2,2})},
                        {"S0",std::make_shared<Tensor>("S0",TensorShape{2,2})},
                        {"S1",std::make_shared<Tensor>("S1",TensorShape{2,2,2})},
                        {"S2",std::make_shared<Tensor>("S2",TensorShape{2,2})}
                       }
                      );
 network.printIt();
 //Remove tensor #6 to create the optimization environment for MPS tensor S1:
 network.deleteTensor(6);
 network.printIt();
}


TEST(NumericsTester, checkSharedTensorNetworkSymbolic)
{
 //3-site MPS closure with 2-body Hamiltonian applied to sites 0 and 1:
 //Z0() = T0(a,b) * T1(b,c,d) * T2(d,e) * H0(a,c,f,g) * S0(f,h) * S1(h,g,i) * S2(i,e)
 // 0      1         2           3         4             5         6           7  <-- tensor id
 auto network = makeSharedTensorNetwork(
                 "{0,1} 3-site MPS closure", //tensor network name
                 "Z0() = T0(a,b) * T1(b,c,d) * T2(d,e) * H0(a,c,f,g) * S0(f,h) * S1(h,g,i) * S2(i,e)", //tensor network specification
                 std::map<std::string,std::shared_ptr<Tensor>>{
                  {"Z0",std::make_shared<Tensor>("Z0")},
                  {"T0",std::make_shared<Tensor>("T0",TensorShape{2,2})},
                  {"T1",std::make_shared<Tensor>("T1",TensorShape{2,2,2})},
                  {"T2",std::make_shared<Tensor>("T2",TensorShape{2,2})},
                  {"H0",std::make_shared<Tensor>("H0",TensorShape{2,2,2,2})},
                  {"S0",std::make_shared<Tensor>("S0",TensorShape{2,2})},
                  {"S1",std::make_shared<Tensor>("S1",TensorShape{2,2,2})},
                  {"S2",std::make_shared<Tensor>("S2",TensorShape{2,2})}
                 }
                );
 network->printIt();
 //Remove tensor #6 to create the optimization environment for MPS tensor S1:
 network->deleteTensor(6);
 network->printIt();
}


TEST(NumericsTester, checkTensorExpansion)
{
 //Building an MPS tensor network with 8 sites and max bond dimension of 6:
 auto & network_build_factory = *(numerics::NetworkBuildFactory::get());
 auto builder = network_build_factory.createNetworkBuilderShared("MPS");
 auto success = builder->setParameter("max_bond_dim",6); assert(success);

 //Tensor network structure:
 //  O-O-O-O-O-O-O-O
 //  | | | | | | | |
 auto output_tensor = makeSharedTensor("Z0",std::vector<DimExtent>{2,2,2,2,2,2,2,2});
 auto network = makeSharedTensorNetwork("TensorTrain",output_tensor,*builder);
 network->printIt();

 //Hamiltonian structure (four 2-body components):
 //  | | | | | | | |
 //  === === === ===
 //  | | | | | | | |
 TensorOperator ham("Hamiltonian"); //will consist of 4 components:
 ham.appendComponent(std::make_shared<Tensor>("H0",TensorShape{2,2,2,2}),
                     {{0,2},{1,3}},            //ket leg map
                     {{0,0},{1,1}},            //bra leg map
                     std::complex<double>{1.0} //expansion coefficient
                    );
 ham.appendComponent(std::make_shared<Tensor>("H1",TensorShape{2,2,2,2}),
                     {{2,2},{3,3}},            //ket leg map
                     {{2,0},{3,1}},            //bra leg map
                     std::complex<double>{1.0} //expansion coefficient
                    );
 ham.appendComponent(std::make_shared<Tensor>("H2",TensorShape{2,2,2,2}),
                     {{4,2},{5,3}},            //ket leg map
                     {{4,0},{5,1}},            //bra leg map
                     std::complex<double>{1.0} //expansion coefficient
                    );
 ham.appendComponent(std::make_shared<Tensor>("H3",TensorShape{2,2,2,2}),
                     {{6,2},{7,3}},            //ket leg map
                     {{6,0},{7,1}},            //bra leg map
                     std::complex<double>{1.0} //expansion coefficient
                    );
 ham.printIt();

 //Application of tensor operator "ham" to the ket tensor network "network":
 //  O-O-O-O-O-O-O-O     O-O-O-O-O-O-O-O
 //  | | | | | | | |     | | | | | | | |
 //  === | | | | | |  +  | | === | | | |  +  ...
 //  | | | | | | | |     | | | | | | | |
 TensorExpansion ket_vector;
 ket_vector.appendComponent(network,std::complex<double>{0.5});
 TensorExpansion oper_times_ket(ket_vector,ham);
 oper_times_ket.printIt();

 //Form the inner product with a conjugated tensor network:
 //  O-O-O-O-O-O-O-O     O-O-O-O-O-O-O-O
 //  | | | | | | | |     | | | | | | | |
 //  === | | | | | |  +  | | === | | | |  +  ...
 //  | | | | | | | |     | | | | | | | |
 //  O-O-O-O-O-O-O-O     O-O-O-O-O-O-O-O
 TensorExpansion bra_vector(ket_vector);
 bra_vector.conjugate();
 bra_vector.printIt();
 TensorExpansion bra_times_oper_times_ket(bra_vector,oper_times_ket);
 bra_times_oper_times_ket.printIt();
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
