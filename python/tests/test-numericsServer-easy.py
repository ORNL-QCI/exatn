import exatn

#Initialize the ExaTN framework
exatn.Initialize()
num_server = exatn.getNumServer()
num_server.createTensor("Z0")
num_server.createTensor("T0", [2,2])
num_server.createTensor("T1", [2,2,2])
num_server.createTensor("T2", [2,2])
num_server.createTensor("H0", [2,2,2,2])
num_server.createTensor("S0", [2,2])
num_server.createTensor("S1", [2,2,2])
num_server.createTensor("S2", [2,2])

num_server.initTensor("Z0",0.0)
num_server.initTensor("T0",0.001)
num_server.initTensor("T1",0.001)
num_server.initTensor("T2",0.001)
num_server.initTensor("H0",0.001)
num_server.initTensor("S0",0.001)
num_server.initTensor("S1",0.001)
num_server.initTensor("S2",0.001)

num_server.evaluateTensorNetwork("{0,1} 3-site MPS closure",
  "Z0() = T0(a,b) * T1(b,c,d) * T2(d,e) * H0(a,c,f,g) * S0(f,h) * S1(h,g,i) * S2(i,e)")

num_server.destroyTensor("Z0")
num_server.destroyTensor("T0")
num_server.destroyTensor("T1")
num_server.destroyTensor("H0")
num_server.destroyTensor("S0")
num_server.destroyTensor("S1")
num_server.destroyTensor("S2")

#Finish up
exatn.Finalize()