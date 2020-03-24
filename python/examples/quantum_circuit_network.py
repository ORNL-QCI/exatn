import sys
from pathlib import Path
sys.path.insert(1, str(Path.home()) + '/.exatn')
import exatn, numpy as np

#Quantum Circuit:
#Q0----H---------
#Q1----H----C----
#Q2----H----N----

#Define the initial qubit state vector:
qzero = np.array([1.0, 0.0], dtype=complex)
hadamard = np.array([[1., 1.],[1., -1.]], dtype=complex)

cnot = np.reshape(np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0]], dtype=complex), (2,2,2,2))

exatn.createTensor('Q0', qzero)
exatn.createTensor('Q1', qzero)
exatn.createTensor('Q2', qzero)

exatn.createTensor('H', hadamard)
exatn.createTensor('CNOT', cnot)

exatn.registerTensorIsometry('H', [0], [1])
exatn.registerTensorIsometry('CNOT', [0,1], [2,3])

circuit = exatn.TensorNetwork('QuantumCircuit')
circuit.appendTensor(1, 'Q0')
circuit.appendTensor(2, 'Q1')
circuit.appendTensor(3, 'Q2')

circuit.appendTensorGate(4, 'H', [0])
circuit.appendTensorGate(5, 'H', [1])
circuit.appendTensorGate(6, 'H', [2])

circuit.appendTensorGate(7, 'CNOT', [1,2])

circuit.printIt()

inverse = exatn.TensorNetwork(circuit)
inverse.rename('InverseCircuit')
inverse.appendTensorGate(8, 'CNOT', [1,2], True)

inverse.appendTensorGate(9, 'H', [2], True)
inverse.appendTensorGate(10, 'H', [1], True)
inverse.appendTensorGate(11, 'H', [0], True)

assert(inverse.collapseIsometries())
inverse.printIt()

assert(exatn.evaluate(circuit))