import sys
from pathlib import Path
sys.path.insert(1, str(Path.home()) + '/.exatn')
import exatn, numpy as np

def test_exatn():
    a1 = np.array([
         [1., 0., 0.],
         [0., 1., 1.]])
    print('A1 shape: ',a1.shape)

    b1 = np.array([
         [ 1., 0.,  3., 0.],
         [ 1., 1.,  2., 2.],
         [-1., 1., -2., 2.]])
    print('B1 shape: ',b1.shape)

    exatn.createTensor('C1', [2, 4], 0.0)
    exatn.createTensor('A1', a1.copy(order='F'))
    exatn.createTensor('B1', b1.copy(order='F'))

    exatn.contractTensors('C1(a,c)=A1(a,b)*B1(b,c)',1.0)

    c1 = exatn.getLocalTensor('C1')
    print('C1 shape: ',c1.shape)

    d1 = np.dot(a1, b1)
    print('D1 shape: ',d1.shape)

    print('NumPy result c1 = a1 * b1:\n', d1)
    print('ExaTN result c1 = a1 * b1:\n', c1)

    exatn.destroyTensor('B1')
    exatn.destroyTensor('A1')
    exatn.destroyTensor('C1')

test_exatn()
