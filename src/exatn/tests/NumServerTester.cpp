#include <gtest/gtest.h>

#include "exatn.hpp"
#include "talshxx.hpp"

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


TEST(NumServerTester, useNumServer)
{
 using exatn::TensorOpCode;
 using exatn::numerics::Tensor;
 using exatn::numerics::TensorShape;
 using exatn::numerics::TensorOperation;
 using exatn::numerics::TensorOpFactory;

 auto & op_factory = *(TensorOpFactory::get()); //tensor operation factory

 //Example of tensor network processing:
 //3-site MPS closure with 2-body Hamiltonian applied to sites 0 and 1:
 //Z0() = T0(a,b) * T1(b,c,d) * T2(d,e) * H0(a,c,f,g) * S0(f,h) * S1(h,g,i) * S2(i,e)
 // 0      1         2           3         4             5         6           7  <-- tensor id

 //Declare participating ExaTN tensors:
 auto z0 = std::make_shared<Tensor>("Z0");
 auto t0 = std::make_shared<Tensor>("T0",TensorShape{2,2});
 auto t1 = std::make_shared<Tensor>("T1",TensorShape{2,2,2});
 auto t2 = std::make_shared<Tensor>("T2",TensorShape{2,2});
 auto h0 = std::make_shared<Tensor>("H0",TensorShape{2,2,2,2});
 auto s0 = std::make_shared<Tensor>("S0",TensorShape{2,2});
 auto s1 = std::make_shared<Tensor>("S1",TensorShape{2,2,2});
 auto s2 = std::make_shared<Tensor>("S2",TensorShape{2,2});

 //Declare a tensor network:
 TensorNetwork network("{0,1} 3-site MPS closure", //tensor network name
  "Z0() = T0(a,b) * T1(b,c,d) * T2(d,e) * H0(a,c,f,g) * S0(f,h) * S1(h,g,i) * S2(i,e)", //tensor network specification
  std::map<std::string,std::shared_ptr<Tensor>>{
   {z0->getName(),z0}, {t0->getName(),t0}, {t1->getName(),t1}, {t2->getName(),t2},
   {h0->getName(),h0}, {s0->getName(),s0}, {s1->getName(),s1}, {s2->getName(),s2}
  }
 );
 network.printIt();

 //Create participating ExaTN tensors:
 std::shared_ptr<TensorOperation> create_z0 = op_factory.createTensorOp(TensorOpCode::CREATE);
 create_z0->setTensorOperand(z0);
 exatn::numericalServer->submit(create_z0);

 std::shared_ptr<TensorOperation> create_t0 = op_factory.createTensorOp(TensorOpCode::CREATE);
 create_t0->setTensorOperand(t0);
 exatn::numericalServer->submit(create_t0);

 std::shared_ptr<TensorOperation> create_t1 = op_factory.createTensorOp(TensorOpCode::CREATE);
 create_t1->setTensorOperand(t1);
 exatn::numericalServer->submit(create_t1);

 std::shared_ptr<TensorOperation> create_t2 = op_factory.createTensorOp(TensorOpCode::CREATE);
 create_t2->setTensorOperand(t2);
 exatn::numericalServer->submit(create_t2);

 std::shared_ptr<TensorOperation> create_h0 = op_factory.createTensorOp(TensorOpCode::CREATE);
 create_h0->setTensorOperand(h0);
 exatn::numericalServer->submit(create_h0);

 std::shared_ptr<TensorOperation> create_s0 = op_factory.createTensorOp(TensorOpCode::CREATE);
 create_s0->setTensorOperand(s0);
 exatn::numericalServer->submit(create_s0);

 std::shared_ptr<TensorOperation> create_s1 = op_factory.createTensorOp(TensorOpCode::CREATE);
 create_s1->setTensorOperand(s1);
 exatn::numericalServer->submit(create_s1);

 std::shared_ptr<TensorOperation> create_s2 = op_factory.createTensorOp(TensorOpCode::CREATE);
 create_s2->setTensorOperand(s2);
 exatn::numericalServer->submit(create_s2);

 //Initialize participating ExaTN tensors:
 std::shared_ptr<TensorOperation> init_z0 = op_factory.createTensorOp(TensorOpCode::TRANSFORM);
 init_z0->setTensorOperand(z0);
 std::dynamic_pointer_cast<TensorOpTransform>(init_z0)->
  resetFunctor(std::shared_ptr<TensorMethod>(new exatn::numerics::FunctorInitVal(0.0)));
 exatn::numericalServer->submit(init_z0);

 std::shared_ptr<TensorOperation> init_t0 = op_factory.createTensorOp(TensorOpCode::TRANSFORM);
 init_t0->setTensorOperand(t0);
 std::dynamic_pointer_cast<TensorOpTransform>(init_t0)->
  resetFunctor(std::shared_ptr<TensorMethod>(new exatn::numerics::FunctorInitVal(0.001)));
 exatn::numericalServer->submit(init_t0);

 std::shared_ptr<TensorOperation> init_t1 = op_factory.createTensorOp(TensorOpCode::TRANSFORM);
 init_t1->setTensorOperand(t1);
 std::dynamic_pointer_cast<TensorOpTransform>(init_t1)->
  resetFunctor(std::shared_ptr<TensorMethod>(new exatn::numerics::FunctorInitVal(0.001)));
 exatn::numericalServer->submit(init_t1);

 std::shared_ptr<TensorOperation> init_t2 = op_factory.createTensorOp(TensorOpCode::TRANSFORM);
 init_t2->setTensorOperand(t2);
 std::dynamic_pointer_cast<TensorOpTransform>(init_t2)->
  resetFunctor(std::shared_ptr<TensorMethod>(new exatn::numerics::FunctorInitVal(0.001)));
 exatn::numericalServer->submit(init_t2);

 std::shared_ptr<TensorOperation> init_h0 = op_factory.createTensorOp(TensorOpCode::TRANSFORM);
 init_h0->setTensorOperand(h0);
 std::dynamic_pointer_cast<TensorOpTransform>(init_h0)->
  resetFunctor(std::shared_ptr<TensorMethod>(new exatn::numerics::FunctorInitVal(0.001)));
 exatn::numericalServer->submit(init_h0);

 std::shared_ptr<TensorOperation> init_s0 = op_factory.createTensorOp(TensorOpCode::TRANSFORM);
 init_s0->setTensorOperand(s0);
 std::dynamic_pointer_cast<TensorOpTransform>(init_s0)->
  resetFunctor(std::shared_ptr<TensorMethod>(new exatn::numerics::FunctorInitVal(0.001)));
 exatn::numericalServer->submit(init_s0);

 std::shared_ptr<TensorOperation> init_s1 = op_factory.createTensorOp(TensorOpCode::TRANSFORM);
 init_s1->setTensorOperand(s1);
 std::dynamic_pointer_cast<TensorOpTransform>(init_s1)->
  resetFunctor(std::shared_ptr<TensorMethod>(new exatn::numerics::FunctorInitVal(0.001)));
 exatn::numericalServer->submit(init_s1);

 std::shared_ptr<TensorOperation> init_s2 = op_factory.createTensorOp(TensorOpCode::TRANSFORM);
 init_s2->setTensorOperand(s2);
 std::dynamic_pointer_cast<TensorOpTransform>(init_s2)->
  resetFunctor(std::shared_ptr<TensorMethod>(new exatn::numerics::FunctorInitVal(0.001)));
 exatn::numericalServer->submit(init_s2);

 //Evaluate the tensor network:
 exatn::numericalServer->submit(network);
 auto synced = exatn::numericalServer->sync(network,true);
 assert(synced);

 //Retrieve the result:
 auto talsh_tensor = exatn::numericalServer->getLocalTensor(z0);

 //Destroy participating ExaTN tensors:
 std::shared_ptr<TensorOperation> destroy_s2 = op_factory.createTensorOp(TensorOpCode::DESTROY);
 destroy_s2->setTensorOperand(s2);
 exatn::numericalServer->submit(destroy_s2);

 std::shared_ptr<TensorOperation> destroy_s1 = op_factory.createTensorOp(TensorOpCode::DESTROY);
 destroy_s1->setTensorOperand(s1);
 exatn::numericalServer->submit(destroy_s1);

 std::shared_ptr<TensorOperation> destroy_s0 = op_factory.createTensorOp(TensorOpCode::DESTROY);
 destroy_s0->setTensorOperand(s0);
 exatn::numericalServer->submit(destroy_s0);

 std::shared_ptr<TensorOperation> destroy_h0 = op_factory.createTensorOp(TensorOpCode::DESTROY);
 destroy_h0->setTensorOperand(h0);
 exatn::numericalServer->submit(destroy_h0);

 std::shared_ptr<TensorOperation> destroy_t2 = op_factory.createTensorOp(TensorOpCode::DESTROY);
 destroy_t2->setTensorOperand(t2);
 exatn::numericalServer->submit(destroy_t2);

 std::shared_ptr<TensorOperation> destroy_t1 = op_factory.createTensorOp(TensorOpCode::DESTROY);
 destroy_t1->setTensorOperand(t1);
 exatn::numericalServer->submit(destroy_t1);

 std::shared_ptr<TensorOperation> destroy_t0 = op_factory.createTensorOp(TensorOpCode::DESTROY);
 destroy_t0->setTensorOperand(t0);
 exatn::numericalServer->submit(destroy_t0);

 std::shared_ptr<TensorOperation> destroy_z0 = op_factory.createTensorOp(TensorOpCode::DESTROY);
 destroy_z0->setTensorOperand(z0);
 exatn::numericalServer->submit(destroy_z0);
 //Grab a beer!
}


TEST(NumServerTester, easyNumServer)
{
 using exatn::numerics::Tensor;
 using exatn::numerics::TensorShape;
 using exatn::TensorElementType;

 //Example of tensor network processing:
 //3-site MPS closure with 2-body Hamiltonian applied to sites 0 and 1:
 //Z0() = T0(a,b) * T1(b,c,d) * T2(d,e) * H0(a,c,f,g) * S0(f,h) * S1(h,g,i) * S2(i,e)
 // 0      1         2           3         4             5         6           7  <-- tensor id

 //Create tensors:
 auto created = false;
 created = exatn::numericalServer->createTensor("Z0",TensorElementType::REAL64); assert(created);
 created = exatn::numericalServer->createTensor("T0",TensorElementType::REAL64,TensorShape{2,2}); assert(created);
 created = exatn::numericalServer->createTensor("T1",TensorElementType::REAL64,TensorShape{2,2,2}); assert(created);
 created = exatn::numericalServer->createTensor("T2",TensorElementType::REAL64,TensorShape{2,2}); assert(created);
 created = exatn::numericalServer->createTensor("H0",TensorElementType::REAL64,TensorShape{2,2,2,2}); assert(created);
 created = exatn::numericalServer->createTensor("S0",TensorElementType::REAL64,TensorShape{2,2}); assert(created);
 created = exatn::numericalServer->createTensor("S1",TensorElementType::REAL64,TensorShape{2,2,2}); assert(created);
 created = exatn::numericalServer->createTensor("S2",TensorElementType::REAL64,TensorShape{2,2}); assert(created);

 //std::cout << "Z0 tensor element type is " << int(exatn::numericalServer->getTensorElementType("Z0")) << std::endl; //debug

 //Initialize tensors:
 auto initialized = false;
 initialized = exatn::numericalServer->initTensor("Z0",0.0); assert(initialized);
 initialized = exatn::numericalServer->initTensor("T0",0.001); assert(initialized);
 initialized = exatn::numericalServer->initTensor("T1",0.001); assert(initialized);
 initialized = exatn::numericalServer->initTensor("T2",0.001); assert(initialized);
 initialized = exatn::numericalServer->initTensor("H0",0.001); assert(initialized);
 initialized = exatn::numericalServer->initTensor("S0",0.001); assert(initialized);
 initialized = exatn::numericalServer->initTensor("S1",0.001); assert(initialized);
 initialized = exatn::numericalServer->initTensor("S2",0.001); assert(initialized);

 //Evaluate a tensor network:
 auto evaluated = false;
 evaluated = exatn::numericalServer->evaluateTensorNetwork("{0,1} 3-site MPS closure",
  "Z0() = T0(a,b) * T1(b,c,d) * T2(d,e) * H0(a,c,f,g) * S0(f,h) * S1(h,g,i) * S2(i,e)");

 //Sync all operations on Z0:
 auto synced = false;
 synced = exatn::numericalServer->sync("Z0"); assert(synced);

 //Destroy tensors:
 auto destroyed = false;
 destroyed = exatn::numericalServer->destroyTensor("S2"); assert(destroyed);
 destroyed = exatn::numericalServer->destroyTensor("S1"); assert(destroyed);
 destroyed = exatn::numericalServer->destroyTensor("S0"); assert(destroyed);
 destroyed = exatn::numericalServer->destroyTensor("H0"); assert(destroyed);
 destroyed = exatn::numericalServer->destroyTensor("T2"); assert(destroyed);
 destroyed = exatn::numericalServer->destroyTensor("T1"); assert(destroyed);
 destroyed = exatn::numericalServer->destroyTensor("T0"); assert(destroyed);
 destroyed = exatn::numericalServer->destroyTensor("Z0"); assert(destroyed);
 //Grab a beer!
}


TEST(NumServerTester, superEasyNumServer)
{
 using exatn::numerics::Tensor;
 using exatn::numerics::TensorShape;
 using exatn::TensorElementType;

 exatn::resetRuntimeLoggingLevel(0); //debug

 //Example of tensor network processing:
 //3-site MPS closure with 2-body Hamiltonian applied to sites 0 and 1:
 //Z0() = T0(a,b) * T1(b,c,d) * T2(d,e) * H0(a,c,f,g) * S0(f,h) * S1(h,g,i) * S2(i,e)
 // 0      1         2           3         4             5         6           7  <-- tensor id

 //Create tensors:
 auto created = false;
 created = exatn::createTensor("Z0",TensorElementType::REAL64); assert(created);
 created = exatn::createTensor("T0",TensorElementType::REAL64,TensorShape{2,2}); assert(created);
 created = exatn::createTensor("T1",TensorElementType::REAL64,TensorShape{2,2,2}); assert(created);
 created = exatn::createTensor("T2",TensorElementType::REAL64,TensorShape{2,2}); assert(created);
 created = exatn::createTensor("H0",TensorElementType::REAL64,TensorShape{2,2,2,2}); assert(created);
 created = exatn::createTensor("S0",TensorElementType::REAL64,TensorShape{2,2}); assert(created);
 created = exatn::createTensor("S1",TensorElementType::REAL64,TensorShape{2,2,2}); assert(created);
 created = exatn::createTensor("S2",TensorElementType::REAL64,TensorShape{2,2}); assert(created);

 //std::cout << "Z0 tensor element type is " << int(exatn::numericalServer->getTensorElementType("Z0")) << std::endl; //debug

 //Initialize tensors:
 auto initialized = false;
 initialized = exatn::initTensor("Z0",0.0); assert(initialized);
 initialized = exatn::initTensor("T0",0.01); assert(initialized);
 initialized = exatn::initTensor("T1",0.01); assert(initialized);
 initialized = exatn::initTensor("T2",0.01); assert(initialized);
 initialized = exatn::initTensor("H0",0.01); assert(initialized);
 initialized = exatn::initTensor("S0",0.01); assert(initialized);
 initialized = exatn::initTensor("S1",0.01); assert(initialized);
 initialized = exatn::initTensor("S2",0.01); assert(initialized);

 //Evaluate a tensor network:
 auto evaluated = false;
 evaluated = exatn::evaluateTensorNetwork("{0,1} 3-site MPS closure",
  "Z0() = T0(a,b) * T1(b,c,d) * T2(d,e) * H0(a,c,f,g) * S0(f,h) * S1(h,g,i) * S2(i,e)");

 //Sync all operations on Z0:
 auto synced = false;
 synced = exatn::sync("Z0"); assert(synced);

 //Retrieve the result (Z0):
 auto access_granted = false;
 auto talsh_tensor = getLocalTensor("Z0");
 const double * body_ptr;
 access_granted = talsh_tensor->getDataAccessHostConst(&body_ptr); assert(access_granted);
 std::cout << "Final Z0 value = " << *body_ptr << " VS correct value of " << 512e-14 << std::endl;
 body_ptr = nullptr;

 //Destroy tensors:
 auto destroyed = false;
 destroyed = exatn::destroyTensor("S2"); assert(destroyed);
 destroyed = exatn::destroyTensor("S1"); assert(destroyed);
 destroyed = exatn::destroyTensor("S0"); assert(destroyed);
 destroyed = exatn::destroyTensor("H0"); assert(destroyed);
 destroyed = exatn::destroyTensor("T2"); assert(destroyed);
 destroyed = exatn::destroyTensor("T1"); assert(destroyed);
 destroyed = exatn::destroyTensor("T0"); assert(destroyed);
 destroyed = exatn::destroyTensor("Z0"); assert(destroyed);
 //Grab a beer!
}


TEST(NumServerTester, circuitNumServer)
{
 using exatn::numerics::Tensor;
 using exatn::numerics::TensorShape;
 using exatn::numerics::TensorNetwork;
 using exatn::TensorElementType;

 exatn::resetRuntimeLoggingLevel(0); //debug

 //Define the initial qubit state vector:
 std::vector<std::complex<double>> qzero {
  {1.0,0.0}, {0.0,0.0}
 };

 //Define quantum gates:
 std::vector<std::complex<double>> hadamard {
  {1.0,0.0}, {1.0,0.0},
  {1.0,0.0}, {-1.0,0.0}
 };
 std::vector<std::complex<double>> cnot {
  {1.0,0.0}, {0.0,0.0}, {0.0,0.0}, {0.0,0.0},
  {0.0,0.0}, {1.0,0.0}, {0.0,0.0}, {0.0,0.0},
  {0.0,0.0}, {0.0,0.0}, {0.0,0.0}, {1.0,0.0},
  {0.0,0.0}, {0.0,0.0}, {1.0,0.0}, {0.0,0.0}
 };

 //Create qubit tensors:
 auto created = false;
 created = exatn::createTensor("Q0",TensorElementType::COMPLEX64,TensorShape{2}); assert(created);
 created = exatn::createTensor("Q1",TensorElementType::COMPLEX64,TensorShape{2}); assert(created);
 created = exatn::createTensor("Q2",TensorElementType::COMPLEX64,TensorShape{2}); assert(created);

 //Create gate tensors:
 auto registered = false;
 created = exatn::createTensor("H",TensorElementType::COMPLEX64,TensorShape{2,2}); assert(created);
 registered = exatn::registerTensorIsometry("H",{0},{1}); assert(registered);
 created = exatn::createTensor("CNOT",TensorElementType::COMPLEX64,TensorShape{2,2,2,2}); assert(created);
 registered = exatn::registerTensorIsometry("CNOT",{0,1},{2,3}); assert(registered);

 //Initialize qubit tensors to zero state:
 auto initialized = false;
 initialized = exatn::initTensorData("Q0",qzero); assert(initialized);
 initialized = exatn::initTensorData("Q1",qzero); assert(initialized);
 initialized = exatn::initTensorData("Q2",qzero); assert(initialized);

 //Initialize necessary gate tensors:
 initialized = exatn::initTensorData("H",hadamard); assert(initialized);
 initialized = exatn::initTensorData("CNOT",cnot); assert(initialized);

 {//Open a new scope:
  //Build a tensor network from the quantum circuit:
  TensorNetwork circuit("QuantumCircuit");
  auto appended = false;
  appended = circuit.appendTensor(1,exatn::getTensor("Q0"),{}); assert(appended);
  appended = circuit.appendTensor(2,exatn::getTensor("Q1"),{}); assert(appended);
  appended = circuit.appendTensor(3,exatn::getTensor("Q2"),{}); assert(appended);

  appended = circuit.appendTensorGate(4,exatn::getTensor("H"),{0}); assert(appended);
  appended = circuit.appendTensorGate(5,exatn::getTensor("H"),{1}); assert(appended);
  appended = circuit.appendTensorGate(6,exatn::getTensor("H"),{2}); assert(appended);

  appended = circuit.appendTensorGate(7,exatn::getTensor("CNOT"),{1,2}); assert(appended);
  circuit.printIt(); //debug

  //Evaluate the tensor network (quantum circuit):
  auto evaluated = false;
  evaluated = exatn::evaluateSync(circuit); assert(evaluated);

  //Synchronize:
  exatn::sync();
 }

 //Destroy all tensors:
 auto destroyed = false;
 destroyed = exatn::destroyTensor("CNOT"); assert(destroyed);
 destroyed = exatn::destroyTensor("H"); assert(destroyed);
 destroyed = exatn::destroyTensor("Q2"); assert(destroyed);
 destroyed = exatn::destroyTensor("Q1"); assert(destroyed);
 destroyed = exatn::destroyTensor("Q0"); assert(destroyed);

 //Synchronize:
 exatn::sync();
 //Grab a beer!
}


TEST(NumServerTester, HamiltonianNumServer)
{
 using exatn::numerics::Tensor;
 using exatn::numerics::TensorShape;
 using exatn::numerics::TensorSignature;
 using exatn::numerics::TensorNetwork;
 using exatn::numerics::TensorOperator;
 using exatn::numerics::TensorExpansion;
 using exatn::TensorElementType;

 exatn::resetRuntimeLoggingLevel(2); //debug

 //Declare MPS tensors:
 auto q0 = std::make_shared<Tensor>("Q0",TensorShape{2,2});
 auto q1 = std::make_shared<Tensor>("Q1",TensorShape{2,2,4});
 auto q2 = std::make_shared<Tensor>("Q2",TensorShape{4,2,2});
 auto q3 = std::make_shared<Tensor>("Q3",TensorShape{2,2});

 //Declare Hamiltonian tensors:
 auto h01 = std::make_shared<Tensor>("H01",TensorShape{2,2,2,2});
 auto h12 = std::make_shared<Tensor>("H12",TensorShape{2,2,2,2});
 auto h23 = std::make_shared<Tensor>("H23",TensorShape{2,2,2,2});
 auto z0 = std::make_shared<Tensor>("Z0",TensorShape{2,2,2,2});

 //Declare the Hamiltonian operator:
 TensorOperator ham("Hamiltonian");
 bool appended = false;
 appended = ham.appendComponent(h01,{{0,0},{1,1}},{{0,2},{1,3}},{1.0,0.0}); assert(appended);
 appended = ham.appendComponent(h12,{{1,0},{2,1}},{{1,2},{2,3}},{1.0,0.0}); assert(appended);
 appended = ham.appendComponent(h23,{{2,0},{3,1}},{{2,2},{3,3}},{1.0,0.0}); assert(appended);

 //Declare the ket MPS tensor network:
 // Q0----Q1----Q2----Q3
 // |     |     |     |
 auto mps_ket = std::make_shared<TensorNetwork>("MPS",
                 "Z0(i0,i1,i2,i3)+=Q0(i0,j0)*Q1(j0,i1,j1)*Q2(j1,i2,j2)*Q3(j2,i3)",
                 std::map<std::string,std::shared_ptr<Tensor>>{
                  {"Z0",z0}, {"Q0",q0}, {"Q1",q1}, {"Q2",q2}, {"Q3",q3}});

 //Declare the ket tensor network expansion:
 // Q0----Q1----Q2----Q3
 // |     |     |     |
 TensorExpansion ket;
 appended = ket.appendComponent(mps_ket,{1.0,0.0}); assert(appended);
 ket.rename("MPSket");

 //Declare the bra tensor network expansion (conjugated ket):
 // |     |     |     |
 // Q0----Q1----Q2----Q3
 TensorExpansion bra(ket);
 bra.conjugate();
 bra.rename("MPSbra");

 //Declare the operator times ket product tensor expansion:
 // Q0----Q1----Q2----Q3    Q0----Q1----Q2----Q3    Q0----Q1----Q2----Q3
 // |     |     |     |     |     |     |     |     |     |     |     |
 // ==H01==     |     |  +  |     ==H12==     |  +  |     |     ==H23==
 // |     |     |     |     |     |     |     |     |     |     |     |
 TensorExpansion ham_ket(ket,ham);
 ham_ket.rename("HamMPSket");

 //Declare the full closed product tensor expansion (scalar):
 // Q0----Q1----Q2----Q3    Q0----Q1----Q2----Q3    Q0----Q1----Q2----Q3
 // |     |     |     |     |     |     |     |     |     |     |     |
 // ==H01==     |     |  +  |     ==H12==     |  +  |     |     ==H23==   =>  AC0()
 // |     |     |     |     |     |     |     |     |     |     |     |
 // Q0----Q1----Q2----Q3    Q0----Q1----Q2----Q3    Q0----Q1----Q2----Q3
 TensorExpansion closed_prod(ham_ket,bra);
 closed_prod.rename("MPSbraHamMPSket");
 closed_prod.printIt(); //debug

 //Declare the derivative tensor expansion with respect to tensor Q1+:
 // Q0----Q1----Q2----Q3    Q0----Q1----Q2----Q3    Q0----Q1----Q2----Q3
 // |     |     |     |     |     |     |     |     |     |     |     |
 // ==H01==     |     |  +  |     ==H12==     |  +  |     |     ==H23==
 // |     |     |     |     |     |     |     |     |     |     |     |
 // Q0--      --Q2----Q3    Q0--      --Q2----Q3    Q0--      --Q2----Q3
 TensorExpansion deriv_q1(closed_prod,"Q1",true);
 deriv_q1.rename("DerivativeQ1");
 deriv_q1.printIt(); //debug

 {//Numerical evaluation:
  //Create MPS tensors:
  bool created = false;
  created = exatn::createTensorSync(q0,TensorElementType::COMPLEX64); assert(created);
  created = exatn::createTensorSync(q1,TensorElementType::COMPLEX64); assert(created);
  created = exatn::createTensorSync(q2,TensorElementType::COMPLEX64); assert(created);
  created = exatn::createTensorSync(q3,TensorElementType::COMPLEX64); assert(created);

  //Create Hamiltonian tensors:
  created = exatn::createTensorSync(h01,TensorElementType::COMPLEX64); assert(created);
  created = exatn::createTensorSync(h12,TensorElementType::COMPLEX64); assert(created);
  created = exatn::createTensorSync(h23,TensorElementType::COMPLEX64); assert(created);

  //Create the Accumulator tensor for the closed tensor expansion:
  created = exatn::createTensorSync("AC0",TensorElementType::COMPLEX64,TensorShape{}); assert(created);
  auto accumulator0 = exatn::getTensor("AC0");

  //Create the Accumulator tensor for the derivative tensor expansion:
  created = exatn::createTensorSync("AC1",TensorElementType::COMPLEX64,q1->getShape()); assert(created);
  auto accumulator1 = exatn::getTensor("AC1");

  //Initialize all input tensors:
  auto initialized = false;
  initialized = exatn::initTensorSync("Q0",1e-2); assert(initialized);
  initialized = exatn::initTensorSync("Q1",1e-2); assert(initialized);
  initialized = exatn::initTensorSync("Q2",1e-2); assert(initialized);
  initialized = exatn::initTensorSync("Q3",1e-2); assert(initialized);
  initialized = exatn::initTensorSync("H01",1e-2); assert(initialized);
  initialized = exatn::initTensorSync("H12",1e-2); assert(initialized);
  initialized = exatn::initTensorSync("H23",1e-2); assert(initialized);
  initialized = exatn::initTensorSync("AC0",0.0); assert(initialized);
  initialized = exatn::initTensorSync("AC1",0.0); assert(initialized);

  //Evaluate the expectation value:
  bool evaluated = false;
  evaluated = exatn::evaluateSync(closed_prod,accumulator0); assert(evaluated);

  //Evaluate the derivative of the expectation value w.r.t. tensor Q1:
  evaluated = exatn::evaluateSync(deriv_q1,accumulator1); assert(evaluated);

  //Retrieve the expectation values:
  for(auto component = closed_prod.begin(); component != closed_prod.end(); ++component){
   auto talsh_tensor = exatn::getLocalTensor(component->network_->getTensor(0)->getName());
   const std::complex<double> * body_ptr;
   auto access_granted = talsh_tensor->getDataAccessHostConst(&body_ptr); assert(access_granted);
   std::cout << "Component " << component->network_->getTensor(0)->getName() << " expectation value = "
             << *body_ptr << " VS correct value of " << 16.384*(1e-15) << std::endl;
   body_ptr = nullptr;
  }
  auto talsh_tensor = exatn::getLocalTensor("AC0"); //accumulator for the whole tensor expansion
  const std::complex<double> * body_ptr;
  auto access_granted = talsh_tensor->getDataAccessHostConst(&body_ptr); assert(access_granted);
  std::cout << "AC0 expectation value = " << *body_ptr << " VS correct value of " << 3*16.384*(1e-15) << std::endl;
  body_ptr = nullptr;

  //Destroy all tensors:
  bool destroyed = false;
  destroyed = exatn::destroyTensorSync("AC1"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("AC0"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("H23"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("H12"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("H01"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("Q3"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("Q2"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("Q1"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("Q0"); assert(destroyed);

  //Synchronize:
  exatn::sync();
 }
 //Grab a beer!
}


int main(int argc, char **argv) {
  exatn::initialize();

  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();

  exatn::finalize();
  return ret;
}
