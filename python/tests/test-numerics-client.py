#An example test file to show the functionality of the exatn::numerics pybind wrapper

import exatn

#initialize ExaTN framework
print("Initialize ExaTN")
exatn.Initialize()

#Corresponds to NumericsTester.checkSimple
#Construct some dummy tensor classes and check their characteristics
tens_sig_list = [[1,5],[2,3]]
tensor_signature = exatn.TensorSignature(tens_sig_list)

print("The example tensor signature rank is "+str(tensor_signature.getRank()))
print("The example tensor signature Dimension spaceID0 is: "+str(tensor_signature.getDimSpaceId(0)))
print("The example tensor signature Dimension subspaceId1 is: "+str(tensor_signature.getDimSubspaceId(1)))
print("The example tensor signature DimSpaceAttr is: "+str(tensor_signature.getDimSpaceAttr(1)))
print("\n")


shape = exatn.TensorShape([61,32])
shape_rank = shape.getRank()
shape_dimext0 = shape.getDimExtent(0)
shape_dimext1 = shape.getDimExtent(1)
print("The example TensorShape rank is: "+str(shape_rank))
print("The example TensorShape DimExtent[0,1] are: ["+str(shape_dimext0)+","+str(shape_dimext1)+"]")
print("\n")


leg = exatn.TensorLeg(1,4)
print("The tensor leg direction is: " + str(leg.getDirection()))
print("The tensor leg ID is: " + str(leg.getTensorId()))

print("\n\n")

#build an example tensor network
network = exatn.TensorNetwork("{0,1} 3-site MPS closure", exatn.Tensor("Z0"),[])

network.appendTensor(1, exatn.Tensor("T0", [2,2]),
                     [exatn.TensorLeg(4,0), exatn.TensorLeg(2,0)])
network.appendTensor(2, exatn.Tensor("T1", [2,2,2]),
                     [exatn.TensorLeg(1,1), exatn.TensorLeg(4,1), exatn.TensorLeg(3,0)])
network.appendTensor(3, exatn.Tensor("T2", [2,2]),
                     [exatn.TensorLeg(2,2), exatn.TensorLeg(7,1)])
network.appendTensor(4, exatn.Tensor("H0", [2,2,2,2]),
                     [exatn.TensorLeg(1,0), exatn.TensorLeg(2,1),
                      exatn.TensorLeg(5,0), exatn.TensorLeg(6,1)])
network.appendTensor(5, exatn.Tensor("S0", [2,2]),
                     [exatn.TensorLeg(4,2), exatn.TensorLeg(6,0)])
network.appendTensor(6, exatn.Tensor("S1",[2,2,2]),
                     [exatn.TensorLeg(5,1), exatn.TensorLeg(4,3), exatn.TensorLeg(7,0)])
network.appendTensor(7, exatn.Tensor("S2", [2,2]),
                     [exatn.TensorLeg(6,2), exatn.TensorLeg(3,1)])

network.finalize()
network.printIt()



#Corresponds to NumericsTester.checkNumServer
#Create and register VectorSpaces and Subspaces with the exatn::NumServer
space1_id = exatn.createVectorSpace("Space1", 1024)
space2_id = exatn.createVectorSpace("Space2", 2048)

subspace1_id = exatn.createSubspace("S11", "Space1", [13,246])
subspace2_id = exatn.createSubspace("S21", "Space2", [1056,1068])

#Get the spaces to access their characteristics
emptyspace = exatn.getVectorSpace("")
space1 = exatn.getVectorSpace("Space1")
space2 = exatn.getVectorSpace("Space2")
subspace1 = exatn.getSubspace("S11")
subspace2 = exatn.getSubspace("S21")

print("Space1 has dimension "+str(space1.getDimension()))
print("Space2 has name "+space2.getName())

print("Subspace1 has lower bound "+str(subspace1.getLowerBound()))
print("Subspace 2 has ID "+str(subspace2.getRegisteredId()))

#One can also get the NumServer and then get the VectorSpace
#(or other NumServer info) from there
numserver = exatn.getNumServer()
space1_numserver = numserver.getVectorSpace("Space1")
print("Space1 has dimension "+str(space1_numserver.getDimension()))


#Finalize the framework
print("Finalize ExaTN")
exatn.Finalize()


print("Finished python numerics client test")