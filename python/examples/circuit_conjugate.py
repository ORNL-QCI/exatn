import sys
from pathlib import Path
sys.path.insert(1, str(Path.home()) + '/.exatn')
import exatn, numpy as np

qzero = np.array([1.0, 0.0], dtype=complex)
unitary = np.reshape(np.array([1,-1j,-1j,1], dtype=complex), (2,2))

exatn.createTensor('Q0', qzero)
exatn.createTensor('U', unitary)
exatn.registerTensorIsometry('U', [0], [1])


circuit = exatn.TensorNetwork('QuantumCircuit')
circuit.appendTensor(1, 'Q0')
circuit.appendTensorGate(2, 'U', [0])
circuit.printIt()

conj_circuit = exatn.TensorNetwork(circuit)
conj_circuit.rename('ConjugatedCircuit')
conj_circuit.conjugate()
conj_circuit.printIt()

assert(exatn.evaluate(circuit))
assert(exatn.evaluate(conj_circuit))

print(exatn.getLocalTensor(circuit.getTensor(0).getName()))
print(exatn.getLocalTensor(conj_circuit.getTensor(0).getName()))
