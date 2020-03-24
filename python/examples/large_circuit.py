import sys
from pathlib import Path
sys.path.insert(1, str(Path.home()) + '/.exatn')
import exatn, numpy as np

qzero = np.array([1.0, 0.0], dtype=complex)
hadamard = np.array([[1., 1.],[1., -1.]], dtype=complex)

nQubits = 10
[exatn.createTensor('Q'+str(i), qzero) for i in range(nQubits)]

exatn.createTensor('H', hadamard)
exatn.registerTensorIsometry('H', [0], [1])

tensorCounter = 1
circuit = exatn.TensorNetwork('QuantumCircuit')
[circuit.appendTensor(tensorCounter+c, 'Q'+str(i)) for i,c in enumerate(range(nQubits))]
tensorCounter += nQubits

qubitReg = exatn.TensorNetwork(circuit)
qubitReg.rename('QubitKet')

[circuit.appendTensorGate(tensorCounter+c, 'H', [i]) for i, c in enumerate(range(nQubits))]
tensorCounter += nQubits

circuit.printIt()

inverse = exatn.TensorNetwork(circuit)
inverse.rename('InverseCircuit')

[inverse.appendTensorGate(tensorCounter+c, 'H', [nQubits - i - 1], True) for i,c in enumerate(range(nQubits))]
tensorCounter += nQubits

assert(inverse.collapseIsometries())

inverse.printIt()


bra = qubitReg
bra.conjugate()
bra.rename('QubitBra')
pairings = [[i,i] for i in range(nQubits)]
inverse.appendTensorNetwork(bra, pairings)

inverse.printIt()
assert(inverse.getRank() == 0)

assert(exatn.evaluate(inverse))

print(exatn.getLocalTensor(inverse.getTensor(0).getName()))