#include "exatn.hpp"
using namespace exatn;
using namespace exatn::numerics;

 int main() {

 exatn::initialize();
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
 //`Finish

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

exatn::finalize();
 //Grab a beer!
return 0;
}

