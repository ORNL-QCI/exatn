import exatn
import mpi4py
from mpi4py import MPI

# Initialize the framework
exatn.Initialize()

taProlCode = """entry: main
scope main group()
 subspace(): s0=[0:127]
 index(s0): a,b,c,d,i,j,k,l
 H2(a,b,c,d) = method("Hamiltonian")
 T2(a,b,c,d) = {1.0,0.0}
 Z2(a,b,c,d) = {0.0,0.0}
 Z2(a,b,c,d) += H2(i,j,k,l) * T2(c,d,i,j) * T2(a,b,k,l)
 X2() = {0.0,0.0}
 X2() += Z2+(a,b,c,d) * Z2(a,b,c,d)
 save X2: tag("Z2_norm")
 ~X2
 ~Z2
 ~T2
 ~H2
end scope main
"""

# Get the MPI Client
client = exatn.getDriverClient('mpi')

# Send some TAProL to execute, this
# is an asynchronous call
jobId = client.interpretTAProL(taProlCode)
print(jobId)

# ... do any other computation since exatn is asynchronous ...

# Retrive the result of the computation
# given the jobId, this will kick off
# a wait until execution is completed
scalarResult = client.getResults(jobId)

print(scalarResult)

# Shutdown the client, this
# will shutdown the server too
client.shutdown()

# Finalize the framework
exatn.Finalize()