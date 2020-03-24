import sys
from pathlib import Path
sys.path.insert(1, str(Path.home()) + '/.exatn')
import exatn

exatn.createTensor('Z0')
exatn.createTensor('T0', [2,2], .01)
exatn.createTensor('T1', [2,2,2], .01)
exatn.createTensor('T2', [2,2], .01)
exatn.createTensor('H0', [2,2,2,2], .01)
exatn.createTensor('S0', [2,2], .01)
exatn.createTensor('S1', [2,2,2], .01)
exatn.createTensor('S2', [2,2], .01)

exatn.evaluateTensorNetwork('{0,1} 3-site MPS closure', 'Z0() = T0(a,b) * T1(b,c,d) * T2(d,e) * H0(a,c,f,g) * S0(f,h) * S1(h,g,i) * S2(i,e)')

z0 = exatn.getLocalTensor('Z0')
assert(abs(z0 - 5.12e-12) < 1e-12)
print(z0)