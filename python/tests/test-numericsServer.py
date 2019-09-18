# An example test script to show the functionality of the exatn::NumericsServer python wrapping

import exatn

#Initialize the ExaTN framework
exatn.Initialize()

# Get the tensor operation factory
op_factory = exatn.TensorOpFactory.get()

# Make a few tensors:
z0 = exatn.Tensor("Z0")
t0 = exatn.Tensor("T0", exatn.TensorShape([2,2]))
t1 = exatn.Tensor("T1", exatn.TensorShape([2,2,2]))
t2 = exatn.Tensor("T2", exatn.TensorShape([2,2]))
h0 = exatn.Tensor("H0", exatn.TensorShape([2,2,2,2]))
s0 = exatn.Tensor("S0", exatn.TensorShape([2,2]))
s1 = exatn.Tensor("S1", exatn.TensorShape([2,2,2]))
s2 = exatn.Tensor("S2", exatn.TensorShape([2,2]))

# Make a dictionary to feed to the tensor network
tensor_dict = {
    z0.getName() : z0,
    t0.getName() : t0,
    t1.getName() : t1,
    t2.getName() : t2,
    h0.getName() : h0,
    s0.getName() : s0,
    s1.getName() : s1,
    s2.getName() : s2
}
# Declare a tensor network:
network = exatn.TensorNetwork("{0,1} 3-site MPS closure", # network name
                              "Z0() = T0(a,b) * T1(b,c,d) * T2(d,e) * H0(a,c,f,g) * S0(f,h) * S1(h,g,i) * S2(i,e)", # network specification
                              tensor_dict)
network.printIt()

num_server = exatn.getNumServer()

# Create participating ExaTN tensors:
create_z0 = op_factory.createTensorOpShared(exatn.TensorOpCode.CREATE)
create_z0.setTensorOperand(z0)
num_server.submit(create_z0)

create_t0 = op_factory.createTensorOpShared(exatn.TensorOpCode.CREATE)
create_t0.setTensorOperand(t0)
num_server.submit(create_t0)

create_t1 = op_factory.createTensorOpShared(exatn.TensorOpCode.CREATE)
create_t1.setTensorOperand(t1)
num_server.submit(create_t1)

create_t2 = op_factory.createTensorOpShared(exatn.TensorOpCode.CREATE)
create_t2.setTensorOperand(t2)
num_server.submit(create_t2)

create_h0 = op_factory.createTensorOpShared(exatn.TensorOpCode.CREATE)
create_h0.setTensorOperand(h0)
num_server.submit(create_h0)

create_s0 = op_factory.createTensorOpShared(exatn.TensorOpCode.CREATE)
create_s0.setTensorOperand(s0)
num_server.submit(create_s0)

create_s1 = op_factory.createTensorOpShared(exatn.TensorOpCode.CREATE)
create_s1.setTensorOperand(s1)
num_server.submit(create_s1)

create_s2 = op_factory.createTensorOpShared(exatn.TensorOpCode.CREATE)
create_s2.setTensorOperand(s2)
num_server.submit(create_s2)

# Evaluate the tensor network
num_server.submit(network)

# Destroy the tensors
destroy_s2 = op_factory.createTensorOpShared(exatn.TensorOpCode.DESTROY)
destroy_s2.setTensorOperand(s2)
num_server.submit(destroy_s2)

destroy_s1 = op_factory.createTensorOpShared(exatn.TensorOpCode.DESTROY)
destroy_s1.setTensorOperand(s1)
num_server.submit(destroy_s1)

destroy_s0 = op_factory.createTensorOpShared(exatn.TensorOpCode.DESTROY)
destroy_s0.setTensorOperand(s0)
num_server.submit(destroy_s0)

destroy_h0 = op_factory.createTensorOpShared(exatn.TensorOpCode.DESTROY)
destroy_h0.setTensorOperand(h0)
num_server.submit(destroy_h0)

destroy_t2 = op_factory.createTensorOpShared(exatn.TensorOpCode.DESTROY)
destroy_t2.setTensorOperand(t2)
num_server.submit(destroy_t2)

destroy_t1 = op_factory.createTensorOpShared(exatn.TensorOpCode.DESTROY)
destroy_t1.setTensorOperand(t1)
num_server.submit(destroy_t1)

destroy_t0 = op_factory.createTensorOpShared(exatn.TensorOpCode.DESTROY)
destroy_t0.setTensorOperand(t0)
num_server.submit(destroy_t0)

destroy_z0 = op_factory.createTensorOpShared(exatn.TensorOpCode.DESTROY)
destroy_z0.setTensorOperand(z0)
num_server.submit(destroy_z0)

#Finish up
exatn.Finalize()