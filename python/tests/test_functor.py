import exatn, numpy as np

exatn.Initialize()

num_server = exatn.getNumServer()

num_server.createTensor("T0", [2,2])
num_server.createTensor("L", [32,32,32])
num_server.initTensor("T0",0.22)
num_server.initTensor("L",0.11)

num_server.createTensor("T1", [4,4,4], exatn.DataType.complex)

def negate(data):
    data *= -1.
    print(data)
    return

def printTensor(data):
    print(data.shape, '\n', data)
    return

num_server.transformTensor("T0", negate)
num_server.transformTensor("T0", negate)

num_server.transformTensor("L", negate)
num_server.transformTensor("L", negate)

num_server.createTensor("Sx", np.array([[0.,1.],[1.,0.]]))
num_server.transformTensor("Sx", printTensor)

num_server.createTensor("Sy", np.array([[0.,-1.j],[1.j,0.]]))
num_server.transformComplexTensor("Sy", printTensor)

num_server.createTensor("Sz", np.array([[1.,0.],[0.,-1.]]))
num_server.transformTensor("Sz", printTensor)

num_server.transformComplexTensor("T1", printTensor)

num_server.sync("T1", True)

num_server.destroyTensor("T0")
num_server.destroyTensor("L")
num_server.destroyTensor("Sx")
num_server.destroyTensor("Sz")
num_server.destroyTensor("T1")

exatn.Finalize()
