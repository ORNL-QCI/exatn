#An example test file to show the functionality of the exatn::numerics pybind wrapper

import exatn

print("Initialize ExaTN")
exatn.Initialize()

#Corresponds to NumericsTester.checkSimple
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
leg.printIt()
print("\n")


#Corresponds to NumericsTester.checkNumServer
num_server = exatn.NumServer()
space1 = [0]
space1_id = num_server.createVectorSpace("Space1",1024, space1)


print ("Finalize ExaTN")
exatn.Finalize()


print("Finished python numerics client test!")