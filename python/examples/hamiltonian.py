import sys
from pathlib import Path
sys.path.insert(1, str(Path.home()) + '/.exatn')
import exatn, numpy as np
 
# Declare MPS tensors:
exatn.createTensor('Q0', [2,2], 1e-2)
exatn.createTensor('Q1', [2,2,4], 1e-2)
exatn.createTensor('Q2', [4,2,2], 1e-2)
exatn.createTensor('Q3', [2,2], 1e-2)

# Declare Hamiltonian Tensors
exatn.createTensor('H01', [2,2,2,2], 1e-2)
exatn.createTensor('H12', [2,2,2,2], 1e-2)
exatn.createTensor('H23', [2,2,2,2], 1e-2)
exatn.createTensor('Z0', [2,2,2,2], 1e-2)

# Get them as exatn.Tensor
q0 = exatn.getTensor('Q0')
q1 = exatn.getTensor('Q1')
q2 = exatn.getTensor('Q2')
q3 = exatn.getTensor('Q3')
h01 = exatn.getTensor('H01')
h12 = exatn.getTensor('H12')
h23 = exatn.getTensor('H23')
z0 = exatn.getTensor('Z0')

# Declare the Hamiltonian Operator
ham = exatn.TensorOperator('Hamiltonian')
ham.appendComponent(h01, [[0,0],[1,1]], [[0,2],[1,3]], 1.0)
ham.appendComponent(h12, [[1,0],[2,1]], [[1,2],[2,3]], 1.0)
ham.appendComponent(h23, [[2,0],[3,1]], [[2,2],[3,3]], 1.0)

# Declare the ket MPS tensor network:
# Q0----Q1----Q2----Q3
# |     |     |     |
mps_ket = exatn.TensorNetwork('MPS', 'Z0(i0,i1,i2,i3)+=Q0(i0,j0)*Q1(j0,i1,j1)*Q2(j1,i2,j2)*Q3(j2,i3)', {'Z0':z0, 'Q0':q0, 'Q1':q1, 'Q2':q2, 'Q3':q3})

# Declare the ket tensor network expansion:
# Q0----Q1----Q2----Q3
# |     |     |     |
ket = exatn.TensorExpansion()
ket.appendComponent(mps_ket, 1.0)
ket.rename('MPSket')

# Declare the bra tensor network expansion (conjugated ket):
# |     |     |     |
# Q0----Q1----Q2----Q3
bra = exatn.TensorExpansion(ket)
bra.conjugate()
bra.rename('MPSbra')

# Declare the operator times ket product tensor expansion:
# Q0----Q1----Q2----Q3    Q0----Q1----Q2----Q3    Q0----Q1----Q2----Q3
# |     |     |     |     |     |     |     |     |     |     |     |
# ==H01==     |     |  +  |     ==H12==     |  +  |     |     ==H23==
# |     |     |     |     |     |     |     |     |     |     |     |
ham_ket = exatn.TensorExpansion(ket, ham)
ham_ket.rename('HamMPSket')

# Declare the full closed product tensor expansion (scalar):
# Q0----Q1----Q2----Q3    Q0----Q1----Q2----Q3    Q0----Q1----Q2----Q3
# |     |     |     |     |     |     |     |     |     |     |     |
# ==H01==     |     |  +  |     ==H12==     |  +  |     |     ==H23==   =>  AC0()
# |     |     |     |     |     |     |     |     |     |     |     |
# Q0----Q1----Q2----Q3    Q0----Q1----Q2----Q3    Q0----Q1----Q2----Q3
closed_prod = exatn.TensorExpansion(ham_ket, bra)
closed_prod.rename('MPSbraHamMPSket')
closed_prod.printIt()

# Declare the derivative tensor expansion with respect to tensor Q1+:
# Q0----Q1----Q2----Q3    Q0----Q1----Q2----Q3    Q0----Q1----Q2----Q3
# |     |     |     |     |     |     |     |     |     |     |     |
# ==H01==     |     |  +  |     ==H12==     |  +  |     |     ==H23==
# |     |     |     |     |     |     |     |     |     |     |     |
# Q0--      --Q2----Q3    Q0--      --Q2----Q3    Q0--      --Q2----Q3
deriv_q1 = exatn.TensorExpansion(closed_prod, 'Q1', True)
deriv_q1.rename('DerivativeQ1')

# Create the Accumulator tensor for the closed tensor expansion:
exatn.createTensor('AC0')
accumulator0 = exatn.getTensor('AC0')
exatn.createTensor('AC1',[2,2,4], 0.0)
accumulator1 = exatn.getTensor('AC1')

# Evaluate the expectation value:
exatn.evaluate(closed_prod, accumulator0)

# Evaluate the derivative of the expectation value w.r.t. tensor Q1:
exatn.evaluate(deriv_q1, accumulator1)

# Print the expectation values
print(exatn.getLocalTensor('AC0'))
[print(exatn.getLocalTensor(c.network.getTensor(0).getName())) for c in closed_prod]