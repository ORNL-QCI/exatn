import exatn, numpy as np

# Create some numpy arrays
s = np.random.rand(2)
r = np.random.rand(2)

# Create the ExaTN tensors
exatn.createTensor("Z0", exatn.DataType.float64)
exatn.createTensor("S", s)
exatn.createTensor("R", r)

# Print S, R
exatn.print("S")
exatn.print("R")

# Demonstrate transformTensor interface by
# negating the tensor data in S
def negate(data):
    data *= -1.
exatn.transformTensor("S", negate)
exatn.print("S")

# Compute the dot product, store in Z0
exatn.evaluateTensorNetwork('MyTN', 'Z0() = S(a) * R(a)')
exatn.print('Z0')

# Compare to what numpy would have gotten
print(np.dot(-s,r))

# Clean up and destroy
exatn.destroyTensor("S")
exatn.destroyTensor("R")
exatn.destroyTensor("Z0")
