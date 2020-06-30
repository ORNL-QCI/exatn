import sys
from pathlib import Path
sys.path.insert(1, str(Path.home()) + '/.exatn')
import exatn
import numpy as np

# Demonstrate simple tensor network manipulation

exatn.createTensor('X', [2, 2], 0)
exatn.createTensor('Y', [2, 2], 0)
exatn.createTensor('Z', [2, 2], 0)
exatn.initTensorRnd('X')
exatn.initTensorRnd('Y')
exatn.initTensorRnd('Z')

tNet = exatn.TensorNetwork('test')
tNet.appendTensor(1, 'X')
tNet.appendTensor(2, 'Y')
tNet.appendTensor(3, 'Z')
# print tensor network
tNet.printIt()
tNetOriginal = exatn.TensorNetwork(tNet)

# Merge X and Y
pattern = tNet.mergeTensors(1, 2, 4)
print("After merge:")
tNet.printIt()
# Print the generic merge pattern
print(pattern)
# Create the merged tensor
pattern = pattern.replace("D", tNet.getTensor(4).getName())
pattern = pattern.replace("L", "X")
pattern = pattern.replace("R", "Y")
print(pattern)

# Perform calculation
exatn.createTensor(tNet.getTensor(4))
exatn.contractTensors(pattern)

# Evaluate the tensor network (after merging two tensors)
exatn.evaluate(tNet)
# Print root tensor
root = exatn.getLocalTensor(tNet.getTensor(0).getName())
print(root)

# Evaluate the *Original* network to make sure it is the same.
tNetOriginal.printIt()
exatn.evaluate(tNetOriginal)
rootOriginal = exatn.getLocalTensor(tNetOriginal.getTensor(0).getName())
print(rootOriginal)

assert(np.allclose(root, rootOriginal))

