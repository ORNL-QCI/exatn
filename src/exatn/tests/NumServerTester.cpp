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
 initialized = exatn::initTensor("T0",0.001); assert(initialized);
 initialized = exatn::initTensor("T1",0.001); assert(initialized);
 initialized = exatn::initTensor("T2",0.001); assert(initialized);
 initialized = exatn::initTensor("H0",0.001); assert(initialized);
 initialized = exatn::initTensor("S0",0.001); assert(initialized);
 initialized = exatn::initTensor("S1",0.001); assert(initialized);
 initialized = exatn::initTensor("S2",0.001); assert(initialized);

 //Evaluate a tensor network:
 auto evaluated = false;
 evaluated = exatn::evaluateTensorNetwork("{0,1} 3-site MPS closure",
  "Z0() = T0(a,b) * T1(b,c,d) * T2(d,e) * H0(a,c,f,g) * S0(f,h) * S1(h,g,i) * S2(i,e)");

 //Sync all operations on Z0:
 auto synced = false;
 synced = exatn::sync("Z0"); assert(synced);

 //Retrieve the result (Z0):
 auto talsh_tensor = getLocalTensor("Z0");

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


int main(int argc, char **argv) {
  exatn::initialize();

  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();

  exatn::finalize();
  return ret;
}
