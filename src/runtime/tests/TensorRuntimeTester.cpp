/*******************************************************************************
 * Copyright (c) 2019 UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * and Eclipse Distribution License v1.0 which accompanies this
 * distribution. The Eclipse Public License is available at
 * http://www.eclipse.org/legal/epl-v10.html and the Eclipse Distribution
 *License is available at https://eclipse.org/org/documents/edl-v10.php
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *******************************************************************************/
#include <gtest/gtest.h>
#include "exatn.hpp"

TEST(TensorRuntimeTester, checkSimple) {

  using exatn::numerics::Tensor;
  using exatn::numerics::TensorShape;
  using exatn::TensorOpCode;
  using exatn::numerics::TensorOperation;
  using exatn::numerics::TensorOpFactory;

  auto & op_factory = *(TensorOpFactory::get()); //tensor operation factory

  //Declare ExaTN tensors:
  auto tensor0 = std::make_shared<Tensor>("tensor0",TensorShape{16,32,16,32});
  auto tensor1 = std::make_shared<Tensor>("tensor1",TensorShape{16,16,64,64});
  auto tensor2 = std::make_shared<Tensor>("tensor2",TensorShape{32,64,64,32});

  //Create tensor operations for actual tensor construction:
  std::shared_ptr<TensorOperation> create_tensor0 = op_factory.createTensorOp(TensorOpCode::CREATE);
  create_tensor0->setTensorOperand(tensor0);

  std::shared_ptr<TensorOperation> create_tensor1 = op_factory.createTensorOp(TensorOpCode::CREATE);
  create_tensor1->setTensorOperand(tensor1);

  std::shared_ptr<TensorOperation> create_tensor2 = op_factory.createTensorOp(TensorOpCode::CREATE);
  create_tensor2->setTensorOperand(tensor2);

  //Create tensor operation for contracting tensors:
  std::shared_ptr<TensorOperation> contract_tensors = op_factory.createTensorOp(TensorOpCode::CONTRACT);
  contract_tensors->setTensorOperand(tensor0);
  contract_tensors->setTensorOperand(tensor1);
  contract_tensors->setTensorOperand(tensor2);
  contract_tensors->setScalar(0,std::complex<double>{0.5,0.0});
  contract_tensors->setIndexPattern("D(a,b,c,d)+=L(c,a,k,l)*R(d,l,k,b)");

  //Create tensor operations for tensor destruction:
  std::shared_ptr<TensorOperation> destroy_tensor2 = op_factory.createTensorOp(TensorOpCode::DESTROY);
  destroy_tensor2->setTensorOperand(tensor2);

  std::shared_ptr<TensorOperation> destroy_tensor1 = op_factory.createTensorOp(TensorOpCode::DESTROY);
  destroy_tensor1->setTensorOperand(tensor1);

  std::shared_ptr<TensorOperation> destroy_tensor0 = op_factory.createTensorOp(TensorOpCode::DESTROY);
  destroy_tensor0->setTensorOperand(tensor0);

#if 0
  //Execute all tensor operations via numerical server:
  exatn::numericalServer->submit(create_tensor0);
  exatn::numericalServer->submit(create_tensor1);
  exatn::numericalServer->submit(create_tensor2);
  exatn::numericalServer->submit(contract_tensors);
  exatn::numericalServer->submit(destroy_tensor2);
  exatn::numericalServer->submit(destroy_tensor1);
  exatn::numericalServer->submit(destroy_tensor0);
#endif

}

int main(int argc, char **argv) {
  exatn::initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  exatn::finalize();
  return ret;
}
