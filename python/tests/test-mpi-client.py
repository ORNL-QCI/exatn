import exatn
import mpi4py
from mpi4py import MPI

# Initialize the framework
exatn.Initialize()

# Get the MPI Client
client = exatn.getDriverClient('mpi')

# Send some TAProL to execute, this
# is an asynchronous call
jobId = client.sendTAProL('Put TAProL Code Here')
print(jobId)

# ... do any other computation since exatn is asynchronous ...

# Retrive the result of the computation
# given the jobId, this will kick off
# a wait until execution is completed
scalarResult = client.retrieveResults(jobId)

print(scalarResult)

# Shutdown the client, this
# will shutdown the server too
client.shutdown()

# Finalize the framework
exatn.Finalize()