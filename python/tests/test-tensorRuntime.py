# An example test file to show the functionality of the exatn::TensorRunTimeTester
# using the python PyBind wrapper API

import exatn
import numpy as np

#initialize ExaTN framework
print("Initialize ExaTN")
exatn.Initialize()

op_factory = exatn.TensorOpFactory.get()

print("Create the tensors")
tensor0 = exatn.Tensor("tensor0", exatn.TensorShape([16,32,16,32]))
tensor1 = exatn.Tensor("tensor1", exatn.TensorShape([16,16,64,64]))
tensor2 = exatn.Tensor("tensor2", exatn.TensorShape([32,64,64,32]))

print("Create tensor operations for actual tensor construction")
create_tensor0 = op_factory.createTensorOpShared(exatn.TensorOpCode.CREATE)
create_tensor0.setTensorOperand(tensor0)

create_tensor1 = op_factory.createTensorOpShared(exatn.TensorOpCode.CREATE)
create_tensor1.setTensorOperand(tensor1)

create_tensor2 = op_factory.createTensorOpShared(exatn.TensorOpCode.CREATE)
create_tensor2.setTensorOperand(tensor2)

print("Create tensor operation for contracting tensors")
contract_tensors = op_factory.createTensorOpShared(exatn.TensorOpCode.CONTRACT)
contract_tensors.setTensorOperand(tensor0)
contract_tensors.setTensorOperand(tensor1)
contract_tensors.setTensorOperand(tensor2)
contract_tensors.setScalar(0,np.cdouble(0.5j + 0))
contract_tensors.setIndexPattern("D(a,b,c,d)+=L(c,a,k,l)*R(d,l,k,b)")

destroy_tensor2 = op_factory.createTensorOpShared(exatn.TensorOpCode.DESTROY)
destroy_tensor2.setTensorOperand(tensor2)

destroy_tensor1 = op_factory.createTensorOpShared(exatn.TensorOpCode.DESTROY)
destroy_tensor1.setTensorOperand(tensor1)

destroy_tensor0 = op_factory.createTensorOpShared(exatn.TensorOpCode.DESTROY)
destroy_tensor0.setTensorOperand(tensor0)

numserver = exatn.getNumServer()
numserver.submit(create_tensor0)
numserver.submit(create_tensor1)
numserver.submit(create_tensor2)
numserver.submit(contract_tensors)
numserver.submit(destroy_tensor0)
numserver.submit(destroy_tensor1)
numserver.submit(destroy_tensor2)

print("Finalize ExaTN")
exatn.Finalize()