import exatn, numpy as np

exatn.createTensor("Sx", np.array([[0.,1.],[1.,0.]]))
exatn.createTensor("Sy", np.array([[0.,-1.j],[1.j,0.]]))
exatn.print("Sx")
exatn.print("Sy")

def negate(data):
    data *= -1.

exatn.transformTensor("Sx", negate)
exatn.print("Sx")

exatn.transformTensor("Sx", negate)
exatn.print("Sx")

exatn.transformTensor("Sy", negate)
exatn.print("Sy")
