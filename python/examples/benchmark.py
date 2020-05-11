import sys, os, re, time
from pathlib import Path
sys.path.insert(1, str(Path.home()) + '/.exatn')
import exatn
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
inputFile = open(dir_path + '/resources/test-cases.txt', 'r') 
count = 0

# Returns an array of tensor dimension from its string expression
# and a map of indices to dimensions
def getTensorDimArray(tensorStr, varDimMap):
    dimStr = tensorStr[2:-1]
    dimVars = dimStr.split(",")
    result = []
    for var in dimVars:
      result.append(int(varDimMap[var]))  
    return result
 

while True: 
    count += 1
    gFlops = 2.0/1e9  
    # Get next line from file 
    line = inputFile.readline() 
  
    # if line is empty 
    # end of file is reached 
    if not line: 
        break

    expr, dimVars = line.strip().split(" & ")
    # Remove the last ';'
    dimVars = dimVars[:-1]
    dimVarList = dimVars.split("; ")
    vardict = {}
    for var in dimVarList:
        varName, varVal = var.split(":")
        vardict[varName] = varVal
        gFlops *= float(varVal)
    exprSplits = re.split(' = | * ', expr)
    rhsTensor = exprSplits[0]
    lhsTensorOperand1 = exprSplits[1]
    lhsTensorOperand2 = exprSplits[3]
    #print("{}:{}:{}".format(rhsTensor, lhsTensorOperand1, lhsTensorOperand2)) 
    # LHS (result) 'C' tensor
    assert(rhsTensor[0] == "C")
    exatn.createTensor("C", getTensorDimArray(rhsTensor, vardict), 0.0)
    
    # RHS A tensor
    assert(lhsTensorOperand1[0] == "A") 
    exatn.createTensor("A", getTensorDimArray(lhsTensorOperand1, vardict), np.random.normal())
    # RHS B tensor
    assert(lhsTensorOperand2[0] == "B") 
    exatn.createTensor("B", getTensorDimArray(lhsTensorOperand2, vardict), np.random.normal())
    # Convert [] to () (ExaTN convention)
    exatnExpr = expr.replace('[','(').replace(']',')')
    # Evaluate
    start = time.process_time()
    exatn.evaluateTensorNetwork('Test', exatnExpr)
    elapsedTime = time.process_time() - start
    
    # Destroy tensors
    exatn.destroyTensor("A")
    exatn.destroyTensor("B")
    exatn.destroyTensor("C")
    # Calc. GFlops/sec
    gFlops = gFlops/elapsedTime
    print("Test {}: {} ({}) || Time elapsed: {} [sec]; GFlops/sec: {}".format(count, expr, dimVars, elapsedTime, gFlops)) 

inputFile.close() 
