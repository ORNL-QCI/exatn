import exatn, numpy as np

exatn.Initialize()

num_server = exatn.getNumServer()
num_server.createTensor("Sx", np.array([[0.,1.],[1.,0.]]))
num_server.createTensor("Sy", np.array([[0.,-1.j],[1.j,0.]]))
num_server.print("Sx")
num_server.print("Sy")

def negate(data):
    data *= -1.

num_server.transformTensor("Sx", negate)
num_server.print("Sx")

num_server.transformTensor("Sx", negate)
num_server.print("Sx")

num_server.transformTensor("Sy", negate)
num_server.print("Sy")

num_server.destroyTensor("Sx")
num_server.destroyTensor("Sy")

exatn.Finalize()