#include <gtest/gtest.h>

#include "exatn.hpp"
#include "talshxx.hpp"

#ifdef MPI_ENABLED
#include "mpi.h"
#endif

#include <iostream>
#include <ios>
#include <utility>

//Test activation:
/*#define EXATN_TEST0
#define EXATN_TEST1
#define EXATN_TEST2
#define EXATN_TEST3
#define EXATN_TEST4
#define EXATN_TEST5
#define EXATN_TEST6
#define EXATN_TEST7
#define EXATN_TEST8
#define EXATN_TEST9
#define EXATN_TEST10
#define EXATN_TEST11
#define EXATN_TEST12
#define EXATN_TEST13
#define EXATN_TEST14
#define EXATN_TEST15
#define EXATN_TEST16
#define EXATN_TEST17*/
//#define EXATN_TEST18 //buggy
//#define EXATN_TEST19
//#define EXATN_TEST20 //MKL only
//#define EXATN_TEST21
#define EXATN_TEST22


#ifdef EXATN_TEST0
TEST(NumServerTester, PerformanceExaTN)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::TensorNetwork;
 using exatn::TensorOperator;
 using exatn::TensorExpansion;
 using exatn::TensorElementType;

 const exatn::DimExtent DIM = 1024; //1024 for low-end CPU, 3072 for Maxwell, 4096 for Pascal and Volta
 const auto TENS_ELEM_TYPE = TensorElementType::REAL32;

 //exatn::resetLoggingLevel(1,2); //debug

 bool success = true;

 std::cout << "Contractions of rank-2 tensors:" << std::endl;
 //Create tensors:
 success = exatn::createTensor("A",TENS_ELEM_TYPE,TensorShape{DIM,DIM}); assert(success);
 success = exatn::createTensor("B",TENS_ELEM_TYPE,TensorShape{DIM,DIM}); assert(success);
 success = exatn::createTensor("C",TENS_ELEM_TYPE,TensorShape{DIM,DIM}); assert(success);
 success = exatn::createTensor("D",TENS_ELEM_TYPE,TensorShape{DIM,DIM}); assert(success);
 success = exatn::createTensor("E",TENS_ELEM_TYPE,TensorShape{DIM,DIM}); assert(success);
 success = exatn::createTensor("F",TENS_ELEM_TYPE,TensorShape{DIM,DIM}); assert(success);
 success = exatn::createTensor("G",TENS_ELEM_TYPE,TensorShape{DIM,DIM}); assert(success);
 success = exatn::createTensor("H",TENS_ELEM_TYPE,TensorShape{DIM,DIM}); assert(success);
 success = exatn::createTensor("I",TENS_ELEM_TYPE,TensorShape{DIM,DIM}); assert(success);

 //Initialize tensors:
 success = exatn::initTensor("A",1e-4); assert(success);
 success = exatn::initTensor("B",1e-3); assert(success);
 success = exatn::initTensor("C",0.0); assert(success);
 success = exatn::initTensor("D",1e-4); assert(success);
 success = exatn::initTensor("E",1e-3); assert(success);
 success = exatn::initTensor("F",0.0); assert(success);
 success = exatn::initTensor("G",1e-4); assert(success);
 success = exatn::initTensor("H",1e-3); assert(success);
 success = exatn::initTensor("I",0.0); assert(success);

 //Contract tensors (case 1):
 std::cout << " Case 1: C=A*B five times: ";
 exatn::sync();
 auto time_start = exatn::Timer::timeInSecHR();
 success = exatn::contractTensors("C(i,j)+=A(k,i)*B(k,j)",1.0); assert(success);
 success = exatn::contractTensors("C(i,j)+=A(i,k)*B(k,j)",1.0); assert(success);
 success = exatn::contractTensors("C(i,j)+=A(k,i)*B(j,k)",1.0); assert(success);
 success = exatn::contractTensors("C(i,j)+=A(i,k)*B(j,k)",1.0); assert(success);
 success = exatn::contractTensors("C(i,j)+=A(k,i)*B(k,j)",1.0); assert(success);
 exatn::sync();
 auto duration = exatn::Timer::timeInSecHR(time_start);
 std::cout << "Average performance (GFlop/s) = " << 5.0*2.0*double{DIM}*double{DIM}*double{DIM}/duration/1e9 << std::endl;

 //Contract tensors (case 2):
 std::cout << " Case 2: C=A*B | F=D*E | I=G*H pipeline: ";
 exatn::sync();
 time_start = exatn::Timer::timeInSecHR();
 success = exatn::contractTensors("I(i,j)+=G(j,k)*H(i,k)",1.0); assert(success);
 success = exatn::contractTensors("F(i,j)+=D(j,k)*E(i,k)",1.0); assert(success);
 success = exatn::contractTensors("C(i,j)+=A(j,k)*B(i,k)",1.0); assert(success);
 exatn::sync();
 duration = exatn::Timer::timeInSecHR(time_start);
 std::cout << "Average performance (GFlop/s) = " << 3.0*2.0*double{DIM}*double{DIM}*double{DIM}/duration/1e9 << std::endl;

 //Contract tensors (case 3):
 std::cout << " Case 3: I=A*B | I=D*E | I=G*H prefetch: ";
 exatn::sync();
 time_start = exatn::Timer::timeInSecHR();
 success = exatn::contractTensors("I(i,j)+=G(j,k)*H(i,k)",1.0); assert(success);
 success = exatn::contractTensors("I(i,j)+=D(j,k)*E(i,k)",1.0); assert(success);
 success = exatn::contractTensors("I(i,j)+=A(j,k)*B(i,k)",1.0); assert(success);
 exatn::sync();
 duration = exatn::Timer::timeInSecHR(time_start);
 std::cout << "Average performance (GFlop/s) = " << 3.0*2.0*double{DIM}*double{DIM}*double{DIM}/duration/1e9 << std::endl;

 //Destroy tensors:
 success = exatn::destroyTensor("I"); assert(success);
 success = exatn::destroyTensor("H"); assert(success);
 success = exatn::destroyTensor("G"); assert(success);
 success = exatn::destroyTensor("F"); assert(success);
 success = exatn::destroyTensor("E"); assert(success);
 success = exatn::destroyTensor("D"); assert(success);
 success = exatn::destroyTensor("C"); assert(success);
 success = exatn::destroyTensor("B"); assert(success);
 success = exatn::destroyTensor("A"); assert(success);

 exatn::sync();

 //Create tensors:
 success = exatn::createTensor("A",TENS_ELEM_TYPE,TensorShape{DIM,DIM,32ULL}); assert(success);
 success = exatn::createTensor("B",TENS_ELEM_TYPE,TensorShape{DIM,DIM,32ULL}); assert(success);
 success = exatn::createTensor("C",TENS_ELEM_TYPE,TensorShape{DIM,DIM}); assert(success);

 //Initialize tensors:
 success = exatn::initTensor("A",1e-4); assert(success);
 success = exatn::initTensor("B",1e-3); assert(success);
 success = exatn::initTensor("C",0.0); assert(success);

 //Contract tensors:
 std::cout << " Case 4: C=A*B out-of-core large dims: ";
 exatn::sync();
 time_start = exatn::Timer::timeInSecHR();
 success = exatn::contractTensors("C(i,j)+=A(j,k,l)*B(i,k,l)",1.0); assert(success);
 exatn::sync();
 duration = exatn::Timer::timeInSecHR(time_start);
 std::cout << "Average performance (GFlop/s) = " << 2.0*double{DIM}*double{DIM}*double{DIM}*double{32}/duration/1e9 << std::endl;

 //Destroy tensors:
 success = exatn::destroyTensor("C"); assert(success);
 success = exatn::destroyTensor("B"); assert(success);
 success = exatn::destroyTensor("A"); assert(success);

 exatn::sync();

/* REQUIRES at least 48 GB Host RAM:
 //Create tensors:
 success = exatn::createTensor("A",TensorElementType::COMPLEX64,TensorShape{2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2}); assert(success);
 success = exatn::createTensor("B",TensorElementType::COMPLEX64,TensorShape{2,2,2,2,2,2,2,2,2,2,1,2,1,2,1,2,2,2,1,2,2,1,1,2,1,2,2,2,2,2,2,1,2,1,2,1,2,2,2,1}); assert(success);
 success = exatn::createTensor("C",TensorElementType::COMPLEX64,TensorShape{2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,1,2,1,2,1,2,2,1,1,1,1,2,1,1,2,1,2,2,2}); assert(success);

 //Initialize tensors:
 success = exatn::initTensor("A",1e-4); assert(success);
 success = exatn::initTensor("B",1e-3); assert(success);
 success = exatn::initTensor("C",0.0); assert(success);

 //Contract tensors:
 std::cout << " Case 5: C=A*B out-of-core small dims: ";
 exatn::sync();
 time_start = exatn::Timer::timeInSecHR();
 success = exatn::contractTensors("C(c49,c40,c13,c50,c47,c14,c15,c41,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c45,c30,c44,c43,c31,c32,c33,c34,c48,c35,c36,c42,c37,c39,c38,c46)+="
  "A(c49,c40,c13,c50,c62,c47,c14,c15,c63,c41,c64,c65,c16,c66,c67,c68,c69,c17,c18,c19,c70,c71,c72,c73,c74,c75)*"
  "B(c20,c21,c64,c22,c69,c67,c23,c24,c25,c26,c27,c28,c29,c45,c30,c44,c68,c43,c31,c73,c72,c32,c33,c66,c34,c75,c74,c71,c65,c48,c70,c35,c63,c36,c42,c37,c39,c38,c46,c62)",1.0); assert(success);
 exatn::sync();
 duration = exatn::Timer::timeInSecHR(time_start);
 std::cout << "Average performance (GFlop/s) = " << 8.0*1.099512e3/duration << std::endl;

 //Destroy tensors:
 success = exatn::destroyTensor("C"); assert(success);
 success = exatn::destroyTensor("B"); assert(success);
 success = exatn::destroyTensor("A"); assert(success);

 exatn::sync(); */

 std::cout << "Tensor decomposition:" << std::endl;
 //Create tensors:
 success = exatn::createTensor("D",TENS_ELEM_TYPE,TensorShape{32,32,32,1}); assert(success);
 success = exatn::createTensor("L",TENS_ELEM_TYPE,TensorShape{32,32,32}); assert(success);
 success = exatn::createTensor("R",TENS_ELEM_TYPE,TensorShape{32,32,1}); assert(success);

 //Initialize tensors:
 success = exatn::initTensorRnd("D"); assert(success);
 success = exatn::initTensor("L",0.0); assert(success);
 success = exatn::initTensor("R",0.0); assert(success);

 //Normalize tensor D:
 double norm1 = 0.0;
 success = exatn::computeNorm1Sync("D",norm1); assert(success);
 success = exatn::scaleTensor("D",1.0/norm1); assert(success);

 //Decompose tensor D:
 success = exatn::decomposeTensorSVDLRSync("D(u0,u1,u2,u3)=L(u0,c0,u1)*R(u2,c0,u3)"); assert(success);

 //Contract tensor factors back with an opposite sign:
 success = exatn::contractTensors("D(u0,u1,u2,u3)+=L(u0,c0,u1)*R(u2,c0,u3)",-1.0); assert(success);
 success = exatn::computeNorm1Sync("D",norm1); assert(success);
 std::cout << " Final 1-norm of tensor D (should be close to zero) = " << norm1 << std::endl;

 //Destroy tensors:
 success = exatn::destroyTensor("R"); assert(success);
 success = exatn::destroyTensor("L"); assert(success);
 success = exatn::destroyTensor("D"); assert(success);

 //Synchronize ExaTN server:
 exatn::sync();
 exatn::resetLoggingLevel(0,0);
}
#endif

#ifdef EXATN_TEST1
TEST(NumServerTester, ExamplarExaTN)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::TensorNetwork;
 using exatn::TensorOperator;
 using exatn::TensorExpansion;
 using exatn::TensorElementType;

 const exatn::DimExtent OC_RANGE = 30;
 const exatn::DimExtent VI_RANGE = 60;
 const auto TENS_ELEM_TYPE = TensorElementType::REAL32;

 //exatn::resetLoggingLevel(1,2); //debug

 bool success = true;

 //Create different process groups:
 const auto global_rank = exatn::getProcessRank();
 const auto total_processes = exatn::getNumProcesses();
 const auto & all_processes = exatn::getDefaultProcessGroup();
 const auto & current_process = exatn::getCurrentProcessGroup();
 std::shared_ptr<exatn::ProcessGroup> me_plus_next, me_plus_prev;
 if(total_processes > 1){
  int color = global_rank / 2; if(global_rank == (total_processes - 1)) color = -1;
  auto me_plus_next = all_processes.split(color);
  color = (global_rank + 1) / 2; if(global_rank == 0) color = -1;
  auto me_plus_prev = all_processes.split(color);
 }
 exatn::sync();

 //Declare and then create (allocate) a tensor (in two steps):
 auto z2 = exatn::makeSharedTensor("Z2",TensorShape{VI_RANGE,VI_RANGE,OC_RANGE,OC_RANGE}); //declares tensor Z2 with no storage
 success = exatn::createTensor(z2,TENS_ELEM_TYPE); assert(success); //allocates REAL64 storage for tensor Z2

 //Create tensors in one step (with allocated storage):
 success = exatn::createTensor("Y2",TENS_ELEM_TYPE,TensorShape{VI_RANGE,VI_RANGE,OC_RANGE,OC_RANGE}); assert(success);
 success = exatn::createTensor("T2",TENS_ELEM_TYPE,TensorShape{VI_RANGE,VI_RANGE,OC_RANGE,OC_RANGE}); assert(success);
 success = exatn::createTensor("S2",TENS_ELEM_TYPE,TensorShape{VI_RANGE,VI_RANGE,OC_RANGE,OC_RANGE}); assert(success);
 success = exatn::createTensor("H2",TENS_ELEM_TYPE,TensorShape{VI_RANGE,VI_RANGE,VI_RANGE,VI_RANGE}); assert(success);
 success = exatn::createTensor("W2",TENS_ELEM_TYPE,TensorShape{VI_RANGE,VI_RANGE,VI_RANGE,VI_RANGE}); assert(success);
 success = exatn::createTensor("ENERGY",TENS_ELEM_TYPE); assert(success); //just a scalar

 //Initialize tensors to a scalar value:
 success = exatn::initTensor("Z2",0.0); assert(success);
 success = exatn::initTensor("Y2",0.0); assert(success);
 success = exatn::initTensor("T2",1e-4); assert(success);
 success = exatn::initTensor("S2",2e-4); assert(success);
 success = exatn::initTensor("H2",1e-3); assert(success);
 success = exatn::initTensor("W2",2e-3); assert(success);
 success = exatn::initTensor("ENERGY",0.0); assert(success);

 //Perform binary tensor contractions (scaled by a scalar):
 success = exatn::contractTensors("Z2(a,b,i,j)+=T2(d,c,j,i)*H2(c,b,d,a)",0.5); assert(success);
 success = exatn::contractTensors("Y2(a,b,i,j)+=S2(c,d,j,i)*W2(b,d,a,c)",1.0); assert(success);
 success = exatn::contractTensors("ENERGY()+=Z2(a,b,i,j)*Z2(a,b,i,j)",0.25); assert(success);
 success = exatn::contractTensors("ENERGY()+=Y2(a,b,i,j)*Y2(a,b,i,j)",0.25); assert(success);

 //Synchronize ExaTN server:
 exatn::sync();

 //Compute 2-norms (synchronously):
 double norm2 = 0.0;
 success = exatn::computeNorm2Sync("Z2",norm2); assert(success);
 std::cout << "Z2 2-norm = " << std::scientific << norm2 << std::endl << std::flush;
 norm2 = 0.0;
 success = exatn::computeNorm2Sync("Y2",norm2); assert(success);
 std::cout << "Y2 2-norm = " << std::scientific << norm2 << std::endl << std::flush;
 norm2 = 0.0;
 success = exatn::computeNorm2Sync("ENERGY",norm2); assert(success);
 std::cout << "ENERGY 2-norm = " << std::scientific << norm2 << std::endl << std::flush;

 //Retrieve scalar ENERGY:
 auto local_copy = exatn::getLocalTensor("ENERGY"); assert(local_copy);
 const exatn::TensorDataType<TENS_ELEM_TYPE>::value * body_ptr;
 auto access_granted = local_copy->getDataAccessHostConst(&body_ptr); assert(access_granted);
 std::cout << "ENERGY value = " << *body_ptr << " VS correct value of "
           << std::pow(std::pow(double{VI_RANGE},2)*(1e-4)*(1e-3)*0.5,2)*std::pow(double{VI_RANGE},2)*std::pow(double{OC_RANGE},2)*0.25
            + std::pow(std::pow(double{VI_RANGE},2)*(2e-4)*(2e-3)*1.0,2)*std::pow(double{VI_RANGE},2)*std::pow(double{OC_RANGE},2)*0.25
           << std::endl << std::flush;
 body_ptr = nullptr;
 local_copy.reset();

 //Synchronize ExaTN server:
 exatn::sync();

 //Destroy all tensors:
 success = exatn::destroyTensor("ENERGY"); assert(success);
 success = exatn::destroyTensor("W2"); assert(success);
 success = exatn::destroyTensor("H2"); assert(success);
 success = exatn::destroyTensor("S2"); assert(success);
 success = exatn::destroyTensor("T2"); assert(success);
 success = exatn::destroyTensor("Y2"); assert(success);
 success = exatn::destroyTensor("Z2"); assert(success);
 z2.reset();

 //Synchronize ExaTN server:
 exatn::sync(all_processes);
 exatn::resetLoggingLevel(0,0);
}
#endif

#ifdef EXATN_TEST2
TEST(NumServerTester, ParallelExaTN)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::TensorNetwork;
 using exatn::TensorOperator;
 using exatn::TensorExpansion;
 using exatn::TensorElementType;

 //exatn::resetLoggingLevel(1,2); //debug

 bool success = true;

 //For multi-process execution:
 auto process_rank = exatn::getProcessRank(); //global rank of the current process
 exatn::ProcessGroup myself(exatn::getCurrentProcessGroup()); //process group containing only the current process
 exatn::ProcessGroup all_processes(exatn::getDefaultProcessGroup()); //group of all processes

 //All processes: Declare and then separately create a tensor:
 auto z0 = exatn::makeSharedTensor("Z0",TensorShape{16,16,16,16}); //declares a tensor Z0[16,16,16,16] with no storage
 success = exatn::createTensor(z0,TensorElementType::REAL32); assert(success); //allocates storage for tensor Z0

 //All processes: Create tensors in one shot (with storage):
 success = exatn::createTensor("T0",TensorElementType::REAL32,TensorShape{16,16}); assert(success);
 success = exatn::createTensor("T1",TensorElementType::REAL32,TensorShape{32,16,32,32}); assert(success);
 success = exatn::createTensor("T2",TensorElementType::REAL32,TensorShape{32,16,32,32}); assert(success);
 success = exatn::createTensor("T3",TensorElementType::REAL32,TensorShape{32,16,32,32}); assert(success);
 success = exatn::createTensor("T4",TensorElementType::REAL32,TensorShape{32,16,32,32}); assert(success);

 //All processes: Initialize tensors to a scalar value:
 success = exatn::initTensor("Z0",0.0); assert(success);
 success = exatn::initTensor("T0",0.0); assert(success);
 success = exatn::initTensor("T1",0.01); assert(success);
 success = exatn::initTensor("T2",0.001); assert(success);
 success = exatn::initTensor("T3",0.0001); assert(success);
 success = exatn::initTensor("T4",0.00001); assert(success);

 //All processes: Scale a tensor by a scalar:
 success = exatn::scaleTensor("T3",0.5); assert(success);

 //All processes: Accumulate a scaled tensor into another tensor:
 success = exatn::addTensors("T2(i,j,k,l)+=T4(i,j,k,l)",0.25); assert(success);

 //All processes: Contract two tensors (scaled by a scalar) and accumulate the result into another tensor:
 success = exatn::contractTensors("T0(i,j)+=T2(c,i,d,e)*T3(d,j,e,c)",0.125); assert(success);

 //All processes: Evaluate the entire tensor network in one shot with a given memory limit per process:
 std::cout << "Original memory limit per process = " << all_processes.getMemoryLimitPerProcess() << std::endl;
 all_processes.resetMemoryLimitPerProcess(exatn::getMemoryBufferSize()/8);
 std::cout << "Corrected memory limit per process = " << all_processes.getMemoryLimitPerProcess() << std::endl;
 success = exatn::evaluateTensorNetwork(all_processes,"FullyConnectedStar",
           "Z0(i,j,k,l)+=T1(d,i,a,e)*T2(a,j,b,f)*T3(b,k,c,e)*T4(c,l,d,f)");
 assert(success);
 //All processes: Synchronize on the computed output tensor Z0:
 success = exatn::sync(all_processes,"Z0"); assert(success);

 //All processes: Compute 2-norm of Z0 (synchronously):
 double norm2 = 0.0;
 success = exatn::computeNorm2Sync("Z0",norm2); assert(success);
 std::cout << "Z0 2-norm = " << norm2 << std::endl << std::flush;

 //Process 0: Compute 2-norm of Z0 by a tensor contraction (synchronously):
 if(process_rank == 0){
  success = exatn::createTensor("S0",TensorElementType::REAL32); assert(success);
  success = exatn::initTensor("S0",0.0); assert(success);
  success = exatn::contractTensorsSync("S0()+=Z0+(i,j,k,l)*Z0(i,j,k,l)",1.0); assert(success);
 }
 //All processes: Replicate tensor S0 to all processes (synchronously):
 success = exatn::replicateTensorSync(all_processes,"S0",0); assert(success);
 //All processes: Retrive a copy of tensor S0 locally:
 auto talsh_tensor = exatn::getLocalTensor("S0");

 //All processes: Destroy all tensors:
 success = exatn::destroyTensor("S0"); assert(success);
 success = exatn::destroyTensor("T4"); assert(success);
 success = exatn::destroyTensor("T3"); assert(success);
 success = exatn::destroyTensor("T2"); assert(success);
 success = exatn::destroyTensor("T1"); assert(success);
 success = exatn::destroyTensor("T0"); assert(success);
 success = exatn::destroyTensor("Z0"); assert(success);
 z0.reset();

 //All processes: Synchronize ExaTN server:
 success = exatn::sync(all_processes); assert(success);
 exatn::resetLoggingLevel(0,0);
}
#endif

#ifdef EXATN_TEST3
TEST(NumServerTester, checkNumServer)
{
 using exatn::VectorSpace;
 using exatn::Subspace;

 const VectorSpace * space1;
 auto space1_id = exatn::numericalServer->createVectorSpace("Space1",1024,&space1);
 space1->printIt();
 std::cout << std::endl;

 const VectorSpace * space2;
 auto space2_id = exatn::numericalServer->createVectorSpace("Space2",2048,&space2);
 space2->printIt();
 std::cout << std::endl;

 const Subspace * subspace1;
 auto subspace1_id = exatn::numericalServer->createSubspace("S11","Space1",{13,246},&subspace1);
 subspace1->printIt();
 std::cout << std::endl;

 const Subspace * subspace2;
 auto subspace2_id = exatn::numericalServer->createSubspace("S21","Space2",{1056,1068},&subspace2);
 subspace2->printIt();
 std::cout << std::endl;

 const VectorSpace * space = exatn::numericalServer->getVectorSpace("");
 space->printIt();
 std::cout << std::endl;

 space = exatn::numericalServer->getVectorSpace("Space2");
 space->printIt();
 std::cout << std::endl;

 const Subspace * subspace = exatn::numericalServer->getSubspace("S11");
 subspace->printIt();
 std::cout << std::endl;
}
#endif

#ifdef EXATN_TEST4
TEST(NumServerTester, useNumServer)
{
 using exatn::TensorOpCode;
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorOperation;
 using exatn::TensorOpFactory;
 using exatn::TensorNetwork;

 auto & op_factory = *(TensorOpFactory::get()); //tensor operation factory

 //Example of tensor network processing:
 //3-site MPS closure with 2-body Hamiltonian applied to sites 0 and 1:
 //Z0() = T0(a,b) * T1(b,c,d) * T2(d,e) * H0(a,c,f,g) * S0(f,h) * S1(h,g,i) * S2(i,e)
 // 0      1         2           3         4             5         6           7  <-- tensor id

 //Declare participating ExaTN tensors:
 auto z0 = std::make_shared<Tensor>("Z0");
 auto t0 = std::make_shared<Tensor>("T0",TensorShape{2,2});
 auto t1 = std::make_shared<Tensor>("T1",TensorShape{2,2,2});
 auto t2 = std::make_shared<Tensor>("T2",TensorShape{2,2});
 auto h0 = std::make_shared<Tensor>("H0",TensorShape{2,2,2,2});
 auto s0 = std::make_shared<Tensor>("S0",TensorShape{2,2});
 auto s1 = std::make_shared<Tensor>("S1",TensorShape{2,2,2});
 auto s2 = std::make_shared<Tensor>("S2",TensorShape{2,2});

 //Declare a tensor network:
 TensorNetwork network("{0,1} 3-site MPS closure", //tensor network name
  "Z0() = T0(a,b) * T1(b,c,d) * T2(d,e) * H0(a,c,f,g) * S0(f,h) * S1(h,g,i) * S2(i,e)", //tensor network specification
  std::map<std::string,std::shared_ptr<Tensor>>{
   {z0->getName(),z0}, {t0->getName(),t0}, {t1->getName(),t1}, {t2->getName(),t2},
   {h0->getName(),h0}, {s0->getName(),s0}, {s1->getName(),s1}, {s2->getName(),s2}
  }
 );
 network.printIt();

 //Create participating ExaTN tensors:
 std::shared_ptr<TensorOperation> create_z0 = op_factory.createTensorOp(TensorOpCode::CREATE);
 create_z0->setTensorOperand(z0);
 exatn::numericalServer->submit(create_z0);

 std::shared_ptr<TensorOperation> create_t0 = op_factory.createTensorOp(TensorOpCode::CREATE);
 create_t0->setTensorOperand(t0);
 exatn::numericalServer->submit(create_t0);

 std::shared_ptr<TensorOperation> create_t1 = op_factory.createTensorOp(TensorOpCode::CREATE);
 create_t1->setTensorOperand(t1);
 exatn::numericalServer->submit(create_t1);

 std::shared_ptr<TensorOperation> create_t2 = op_factory.createTensorOp(TensorOpCode::CREATE);
 create_t2->setTensorOperand(t2);
 exatn::numericalServer->submit(create_t2);

 std::shared_ptr<TensorOperation> create_h0 = op_factory.createTensorOp(TensorOpCode::CREATE);
 create_h0->setTensorOperand(h0);
 exatn::numericalServer->submit(create_h0);

 std::shared_ptr<TensorOperation> create_s0 = op_factory.createTensorOp(TensorOpCode::CREATE);
 create_s0->setTensorOperand(s0);
 exatn::numericalServer->submit(create_s0);

 std::shared_ptr<TensorOperation> create_s1 = op_factory.createTensorOp(TensorOpCode::CREATE);
 create_s1->setTensorOperand(s1);
 exatn::numericalServer->submit(create_s1);

 std::shared_ptr<TensorOperation> create_s2 = op_factory.createTensorOp(TensorOpCode::CREATE);
 create_s2->setTensorOperand(s2);
 exatn::numericalServer->submit(create_s2);

 //Initialize participating ExaTN tensors:
 std::shared_ptr<TensorOperation> init_z0 = op_factory.createTensorOp(TensorOpCode::TRANSFORM);
 init_z0->setTensorOperand(z0);
 std::dynamic_pointer_cast<exatn::numerics::TensorOpTransform>(init_z0)->
  resetFunctor(std::shared_ptr<exatn::TensorMethod>(new exatn::numerics::FunctorInitVal(0.0)));
 exatn::numericalServer->submit(init_z0);

 std::shared_ptr<TensorOperation> init_t0 = op_factory.createTensorOp(TensorOpCode::TRANSFORM);
 init_t0->setTensorOperand(t0);
 std::dynamic_pointer_cast<exatn::numerics::TensorOpTransform>(init_t0)->
  resetFunctor(std::shared_ptr<exatn::TensorMethod>(new exatn::numerics::FunctorInitVal(0.001)));
 exatn::numericalServer->submit(init_t0);

 std::shared_ptr<TensorOperation> init_t1 = op_factory.createTensorOp(TensorOpCode::TRANSFORM);
 init_t1->setTensorOperand(t1);
 std::dynamic_pointer_cast<exatn::numerics::TensorOpTransform>(init_t1)->
  resetFunctor(std::shared_ptr<exatn::TensorMethod>(new exatn::numerics::FunctorInitVal(0.001)));
 exatn::numericalServer->submit(init_t1);

 std::shared_ptr<TensorOperation> init_t2 = op_factory.createTensorOp(TensorOpCode::TRANSFORM);
 init_t2->setTensorOperand(t2);
 std::dynamic_pointer_cast<exatn::numerics::TensorOpTransform>(init_t2)->
  resetFunctor(std::shared_ptr<exatn::TensorMethod>(new exatn::numerics::FunctorInitVal(0.001)));
 exatn::numericalServer->submit(init_t2);

 std::shared_ptr<TensorOperation> init_h0 = op_factory.createTensorOp(TensorOpCode::TRANSFORM);
 init_h0->setTensorOperand(h0);
 std::dynamic_pointer_cast<exatn::numerics::TensorOpTransform>(init_h0)->
  resetFunctor(std::shared_ptr<exatn::TensorMethod>(new exatn::numerics::FunctorInitVal(0.001)));
 exatn::numericalServer->submit(init_h0);

 std::shared_ptr<TensorOperation> init_s0 = op_factory.createTensorOp(TensorOpCode::TRANSFORM);
 init_s0->setTensorOperand(s0);
 std::dynamic_pointer_cast<exatn::numerics::TensorOpTransform>(init_s0)->
  resetFunctor(std::shared_ptr<exatn::TensorMethod>(new exatn::numerics::FunctorInitVal(0.001)));
 exatn::numericalServer->submit(init_s0);

 std::shared_ptr<TensorOperation> init_s1 = op_factory.createTensorOp(TensorOpCode::TRANSFORM);
 init_s1->setTensorOperand(s1);
 std::dynamic_pointer_cast<exatn::numerics::TensorOpTransform>(init_s1)->
  resetFunctor(std::shared_ptr<exatn::TensorMethod>(new exatn::numerics::FunctorInitVal(0.001)));
 exatn::numericalServer->submit(init_s1);

 std::shared_ptr<TensorOperation> init_s2 = op_factory.createTensorOp(TensorOpCode::TRANSFORM);
 init_s2->setTensorOperand(s2);
 std::dynamic_pointer_cast<exatn::numerics::TensorOpTransform>(init_s2)->
  resetFunctor(std::shared_ptr<exatn::TensorMethod>(new exatn::numerics::FunctorInitVal(0.001)));
 exatn::numericalServer->submit(init_s2);

 //Evaluate the tensor network:
 exatn::numericalServer->submit(network);
 auto synced = exatn::numericalServer->sync(network,true); assert(synced);

 //Print the tensor network in a symbolic form:
 std::string network_printed;
 auto printed = network.printTensorNetwork(network_printed); assert(printed);
 std::cout << "Reconstructed symbolic tensor network: " << network_printed;

 //Retrieve the result:
 auto talsh_tensor = exatn::numericalServer->getLocalTensor(z0);

 //Destroy participating ExaTN tensors:
 std::shared_ptr<TensorOperation> destroy_s2 = op_factory.createTensorOp(TensorOpCode::DESTROY);
 destroy_s2->setTensorOperand(s2);
 exatn::numericalServer->submit(destroy_s2);

 std::shared_ptr<TensorOperation> destroy_s1 = op_factory.createTensorOp(TensorOpCode::DESTROY);
 destroy_s1->setTensorOperand(s1);
 exatn::numericalServer->submit(destroy_s1);

 std::shared_ptr<TensorOperation> destroy_s0 = op_factory.createTensorOp(TensorOpCode::DESTROY);
 destroy_s0->setTensorOperand(s0);
 exatn::numericalServer->submit(destroy_s0);

 std::shared_ptr<TensorOperation> destroy_h0 = op_factory.createTensorOp(TensorOpCode::DESTROY);
 destroy_h0->setTensorOperand(h0);
 exatn::numericalServer->submit(destroy_h0);

 std::shared_ptr<TensorOperation> destroy_t2 = op_factory.createTensorOp(TensorOpCode::DESTROY);
 destroy_t2->setTensorOperand(t2);
 exatn::numericalServer->submit(destroy_t2);

 std::shared_ptr<TensorOperation> destroy_t1 = op_factory.createTensorOp(TensorOpCode::DESTROY);
 destroy_t1->setTensorOperand(t1);
 exatn::numericalServer->submit(destroy_t1);

 std::shared_ptr<TensorOperation> destroy_t0 = op_factory.createTensorOp(TensorOpCode::DESTROY);
 destroy_t0->setTensorOperand(t0);
 exatn::numericalServer->submit(destroy_t0);

 std::shared_ptr<TensorOperation> destroy_z0 = op_factory.createTensorOp(TensorOpCode::DESTROY);
 destroy_z0->setTensorOperand(z0);
 exatn::numericalServer->submit(destroy_z0);
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST5
TEST(NumServerTester, easyNumServer)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorElementType;

 //Example of tensor network processing:
 //3-site MPS closure with 2-body Hamiltonian applied to sites 0 and 1:
 //Z0() = T0(a,b) * T1(b,c,d) * T2(d,e) * H0(a,c,f,g) * S0(f,h) * S1(h,g,i) * S2(i,e)
 // 0      1         2           3         4             5         6           7  <-- tensor id

 //Create tensors:
 auto created = false;
 created = exatn::numericalServer->createTensor("Z0",TensorElementType::REAL64); assert(created);
 created = exatn::numericalServer->createTensor("T0",TensorElementType::REAL64,TensorShape{2,2}); assert(created);
 created = exatn::numericalServer->createTensor("T1",TensorElementType::REAL64,TensorShape{2,2,2}); assert(created);
 created = exatn::numericalServer->createTensor("T2",TensorElementType::REAL64,TensorShape{2,2}); assert(created);
 created = exatn::numericalServer->createTensor("H0",TensorElementType::REAL64,TensorShape{2,2,2,2}); assert(created);
 created = exatn::numericalServer->createTensor("S0",TensorElementType::REAL64,TensorShape{2,2}); assert(created);
 created = exatn::numericalServer->createTensor("S1",TensorElementType::REAL64,TensorShape{2,2,2}); assert(created);
 created = exatn::numericalServer->createTensor("S2",TensorElementType::REAL64,TensorShape{2,2}); assert(created);

 //std::cout << "Z0 tensor element type is " << int(exatn::numericalServer->getTensorElementType("Z0")) << std::endl; //debug

 //Initialize tensors:
 auto initialized = false;
 initialized = exatn::numericalServer->initTensor("Z0",0.0); assert(initialized);
 initialized = exatn::numericalServer->initTensor("T0",0.001); assert(initialized);
 initialized = exatn::numericalServer->initTensor("T1",0.001); assert(initialized);
 initialized = exatn::numericalServer->initTensor("T2",0.001); assert(initialized);
 initialized = exatn::numericalServer->initTensor("H0",0.001); assert(initialized);
 initialized = exatn::numericalServer->initTensor("S0",0.001); assert(initialized);
 initialized = exatn::numericalServer->initTensor("S1",0.001); assert(initialized);
 initialized = exatn::numericalServer->initTensor("S2",0.001); assert(initialized);

 //Evaluate a tensor network:
 auto evaluated = false;
 evaluated = exatn::numericalServer->evaluateTensorNetwork("{0,1} 3-site MPS closure",
  "Z0() = T0(a,b) * T1(b,c,d) * T2(d,e) * H0(a,c,f,g) * S0(f,h) * S1(h,g,i) * S2(i,e)");

 //Sync all operations on Z0:
 auto synced = false;
 synced = exatn::numericalServer->sync("Z0"); assert(synced);

 //Destroy tensors:
 auto destroyed = false;
 destroyed = exatn::numericalServer->destroyTensor("S2"); assert(destroyed);
 destroyed = exatn::numericalServer->destroyTensor("S1"); assert(destroyed);
 destroyed = exatn::numericalServer->destroyTensor("S0"); assert(destroyed);
 destroyed = exatn::numericalServer->destroyTensor("H0"); assert(destroyed);
 destroyed = exatn::numericalServer->destroyTensor("T2"); assert(destroyed);
 destroyed = exatn::numericalServer->destroyTensor("T1"); assert(destroyed);
 destroyed = exatn::numericalServer->destroyTensor("T0"); assert(destroyed);
 destroyed = exatn::numericalServer->destroyTensor("Z0"); assert(destroyed);
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST6
TEST(NumServerTester, superEasyNumServer)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorElementType;

 //exatn::resetLoggingLevel(1,2); //debug

 //Example of tensor network processing:
 //3-site MPS closure with 2-body Hamiltonian applied to sites 0 and 1:
 //Z0() = T0(a,b) * T1(b,c,d) * T2(d,e) * H0(a,c,f,g) * S0(f,h) * S1(h,g,i) * S2(i,e)
 // 0      1         2           3         4             5         6           7  <-- tensor id

 //Create tensors:
 auto created = false;
 created = exatn::createTensor("Z0",TensorElementType::REAL64); assert(created);
 created = exatn::createTensor("T0",TensorElementType::REAL64,TensorShape{2,2}); assert(created);
 created = exatn::createTensor("T1",TensorElementType::REAL64,TensorShape{2,2,2}); assert(created);
 created = exatn::createTensor("T2",TensorElementType::REAL64,TensorShape{2,2}); assert(created);
 created = exatn::createTensor("H0",TensorElementType::REAL64,TensorShape{2,2,2,2}); assert(created);
 created = exatn::createTensor("S0",TensorElementType::REAL64,TensorShape{2,2}); assert(created);
 created = exatn::createTensor("S1",TensorElementType::REAL64,TensorShape{2,2,2}); assert(created);
 created = exatn::createTensor("S2",TensorElementType::REAL64,TensorShape{2,2}); assert(created);

 //std::cout << "Z0 tensor element type is " << int(exatn::numericalServer->getTensorElementType("Z0")) << std::endl; //debug

 //Initialize tensors:
 auto initialized = false;
 initialized = exatn::initTensor("Z0",0.0); assert(initialized);
 initialized = exatn::initTensor("T0",0.01); assert(initialized);
 initialized = exatn::initTensor("T1",0.01); assert(initialized);
 initialized = exatn::initTensor("T2",0.01); assert(initialized);
 initialized = exatn::initTensor("H0",0.01); assert(initialized);
 initialized = exatn::initTensor("S0",0.01); assert(initialized);
 initialized = exatn::initTensor("S1",0.01); assert(initialized);
 initialized = exatn::initTensor("S2",0.01); assert(initialized);

 //Evaluate a tensor network:
 auto evaluated = false;
 evaluated = exatn::evaluateTensorNetwork("{0,1} 3-site MPS closure",
  "Z0() = T0(a,b) * T1(b,c,d) * T2(d,e) * H0(a,c,f,g) * S0(f,h) * S1(h,g,i) * S2(i,e)");

 //Sync all operations on Z0:
 auto synced = false;
 synced = exatn::sync("Z0"); assert(synced);

 //Retrieve the result (Z0):
 auto access_granted = false;
 auto talsh_tensor = exatn::getLocalTensor("Z0");
 const double * body_ptr;
 access_granted = talsh_tensor->getDataAccessHostConst(&body_ptr); assert(access_granted);
 std::cout << "Final Z0 value = " << *body_ptr << " VS correct value of " << 512e-14 << std::endl;
 body_ptr = nullptr;

 //Destroy tensors:
 auto destroyed = false;
 destroyed = exatn::destroyTensor("S2"); assert(destroyed);
 destroyed = exatn::destroyTensor("S1"); assert(destroyed);
 destroyed = exatn::destroyTensor("S0"); assert(destroyed);
 destroyed = exatn::destroyTensor("H0"); assert(destroyed);
 destroyed = exatn::destroyTensor("T2"); assert(destroyed);
 destroyed = exatn::destroyTensor("T1"); assert(destroyed);
 destroyed = exatn::destroyTensor("T0"); assert(destroyed);
 destroyed = exatn::destroyTensor("Z0"); assert(destroyed);
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST7
TEST(NumServerTester, circuitNumServer)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorNetwork;
 using exatn::TensorElementType;

 //exatn::resetLoggingLevel(1,2); //debug

 //Quantum Circuit:
 //Q0----H---------
 //Q1----H----C----
 //Q2----H----N----

 //Define the initial qubit state vector:
 std::vector<std::complex<double>> qzero {
  {1.0,0.0}, {0.0,0.0}
 };

 //Define quantum gates:
 std::vector<std::complex<double>> hadamard {
  {1.0,0.0}, {1.0,0.0},
  {1.0,0.0}, {-1.0,0.0}
 };
 std::vector<std::complex<double>> cnot {
  {1.0,0.0}, {0.0,0.0}, {0.0,0.0}, {0.0,0.0},
  {0.0,0.0}, {1.0,0.0}, {0.0,0.0}, {0.0,0.0},
  {0.0,0.0}, {0.0,0.0}, {0.0,0.0}, {1.0,0.0},
  {0.0,0.0}, {0.0,0.0}, {1.0,0.0}, {0.0,0.0}
 };

 //Create qubit tensors:
 auto created = false;
 created = exatn::createTensor("Q0",TensorElementType::COMPLEX64,TensorShape{2}); assert(created);
 created = exatn::createTensor("Q1",TensorElementType::COMPLEX64,TensorShape{2}); assert(created);
 created = exatn::createTensor("Q2",TensorElementType::COMPLEX64,TensorShape{2}); assert(created);

 //Create gate tensors:
 auto registered = false;
 created = exatn::createTensor("H",TensorElementType::COMPLEX64,TensorShape{2,2}); assert(created);
 registered = exatn::registerTensorIsometry("H",{0},{1}); assert(registered);
 created = exatn::createTensor("CNOT",TensorElementType::COMPLEX64,TensorShape{2,2,2,2}); assert(created);
 registered = exatn::registerTensorIsometry("CNOT",{0,1},{2,3}); assert(registered);

 //Initialize qubit tensors to zero state:
 auto initialized = false;
 initialized = exatn::initTensorData("Q0",qzero); assert(initialized);
 initialized = exatn::initTensorData("Q1",qzero); assert(initialized);
 initialized = exatn::initTensorData("Q2",qzero); assert(initialized);

 //Initialize necessary gate tensors:
 initialized = exatn::initTensorData("H",hadamard); assert(initialized);
 initialized = exatn::initTensorData("CNOT",cnot); assert(initialized);

 {//Open a new scope:
  //Build a tensor network from the quantum circuit:
  TensorNetwork circuit("QuantumCircuit");
  auto appended = false;
  appended = circuit.appendTensor(1,exatn::getTensor("Q0"),{}); assert(appended);
  appended = circuit.appendTensor(2,exatn::getTensor("Q1"),{}); assert(appended);
  appended = circuit.appendTensor(3,exatn::getTensor("Q2"),{}); assert(appended);

  appended = circuit.appendTensorGate(4,exatn::getTensor("H"),{0}); assert(appended);
  appended = circuit.appendTensorGate(5,exatn::getTensor("H"),{1}); assert(appended);
  appended = circuit.appendTensorGate(6,exatn::getTensor("H"),{2}); assert(appended);

  appended = circuit.appendTensorGate(7,exatn::getTensor("CNOT"),{1,2}); assert(appended);
  circuit.printIt(); //debug

  //Contract the circuit tensor network with its conjugate:
  TensorNetwork inverse(circuit);
  inverse.rename("InverseCircuit");
  appended = inverse.appendTensorGate(8,exatn::getTensor("CNOT"),{1,2},true); assert(appended);
  appended = inverse.appendTensorGate(9,exatn::getTensor("H"),{2},true); assert(appended);
  appended = inverse.appendTensorGate(10,exatn::getTensor("H"),{1},true); assert(appended);
  appended = inverse.appendTensorGate(11,exatn::getTensor("H"),{0},true); assert(appended);
  auto collapsed = inverse.collapseIsometries(); assert(collapsed);
  inverse.printIt(); //debug

  //Evaluate the quantum circuit expressed as a tensor network:
  auto evaluated = false;
  evaluated = exatn::evaluateSync(circuit); assert(evaluated);

  //Synchronize:
  exatn::sync();
 }

 //Destroy all tensors:
 auto destroyed = false;
 destroyed = exatn::destroyTensor("CNOT"); assert(destroyed);
 destroyed = exatn::destroyTensor("H"); assert(destroyed);
 destroyed = exatn::destroyTensor("Q2"); assert(destroyed);
 destroyed = exatn::destroyTensor("Q1"); assert(destroyed);
 destroyed = exatn::destroyTensor("Q0"); assert(destroyed);

 //Synchronize:
 exatn::sync();
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST8
TEST(NumServerTester, circuitConjugateNumServer)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorNetwork;
 using exatn::TensorElementType;

 //exatn::resetLoggingLevel(1,2); //debug

 //Define the initial qubit state vector:
 std::vector<std::complex<double>> qzero {
  {1.0,0.0}, {0.0,0.0}
 };

 //Define quantum gates: *NEGATIVE* imaginary
 std::vector<std::complex<double>> unitary {
  {1.0, 0.0}, {0.0,-1.0},
  {0.0,-1.0}, {1.0, 0.0}
 };

 //Create tensors:
 bool created = exatn::createTensor("Q0", TensorElementType::COMPLEX64,TensorShape{2}); assert(created);
 created = exatn::createTensor("U", TensorElementType::COMPLEX64, TensorShape{2,2}); assert(created);
 bool registered = exatn::registerTensorIsometry("U", {0}, {1}); assert(registered);

 //Initialize tensors:
 bool initialized = exatn::initTensorData("Q0", qzero); assert(initialized);
 initialized = exatn::initTensorData("U", unitary); assert(initialized);

 {//Open a new scope:
  //Build a tensor network representing the quantum circuit:
  TensorNetwork circuit("QuantumCircuit");
  bool appended = circuit.appendTensor(1, exatn::getTensor("Q0"), {}); assert(appended);
  appended = circuit.appendTensorGate(2, exatn::getTensor("U"), {0}); assert(appended);
  circuit.printIt(); //debug

  //Build a conjugated tensor network:
  TensorNetwork conj_circuit(circuit);
  conj_circuit.rename("ConjugatedCircuit");
  conj_circuit.conjugate();
  conj_circuit.printIt(); //debug

  bool evaluated = exatn::evaluateSync(circuit); assert(evaluated);
  evaluated = exatn::evaluateSync(conj_circuit); assert(evaluated);

  //Synchronize:
  exatn::sync();

  //Retrieve the results:
  auto talsh_tensor0 = exatn::getLocalTensor(circuit.getTensor(0)->getName());
  const std::complex<double> * body_ptr0;
  if(talsh_tensor0->getDataAccessHostConst(&body_ptr0)){
   std::cout << "[";
   for(int i = 0; i < talsh_tensor0->getVolume(); ++i){
    std::cout << body_ptr0[i];
   }
   std::cout << "]\n";
  }

  auto talsh_tensor1 = exatn::getLocalTensor(conj_circuit.getTensor(0)->getName());
  const std::complex<double> * body_ptr1;
  if(talsh_tensor1->getDataAccessHostConst(&body_ptr1)){
   std::cout << "[";
   for(int i = 0; i < talsh_tensor1->getVolume(); ++i){
    std::cout << body_ptr1[i];
   }
   std::cout << "]\n";
  }
 }

 //Destroy tensors:
 bool destroyed = exatn::destroyTensor("U"); assert(destroyed);
 destroyed = exatn::destroyTensor("Q0"); assert(destroyed);

 //Synchronize:
 exatn::sync();
 //Grab a coffee!
}
#endif

#ifdef EXATN_TEST9
TEST(NumServerTester, largeCircuitNumServer)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorNetwork;
 using exatn::TensorElementType;

 //exatn::resetLoggingLevel(1,2); //debug

 //Quantum Circuit:
 //Q00---H-----
 //Q01---H-----
 // |
 //Q49---H-----

 const unsigned int nbQubits = 10;

 //Define the initial qubit state vector:
 std::vector<std::complex<double>> qzero {
  {1.0,0.0}, {0.0,0.0}
 };

 //Define quantum gates:
 std::vector<std::complex<double>> hadamard {
  {1.0,0.0}, {1.0,0.0},
  {1.0,0.0}, {-1.0,0.0}
 };

 //Create qubit tensors:
 for (unsigned int i = 0; i < nbQubits; ++i) {
  const bool created = exatn::createTensor("Q" + std::to_string(i),TensorElementType::COMPLEX64,TensorShape{2});
  assert(created);
 }

 //Create gate tensors:
 {
  const bool created = exatn::createTensor("H",TensorElementType::COMPLEX64,TensorShape{2,2});
  assert(created);
  const bool registered =(exatn::registerTensorIsometry("H",{0},{1}));
  assert(registered);
 }

 //Initialize qubit tensors to zero state:
 for (unsigned int i = 0; i < nbQubits; ++i) {
  const bool initialized = exatn::initTensorData("Q" + std::to_string(i),qzero);
  assert(initialized);
 }

 //Initialize necessary gate tensors:
 {
  const bool initialized = exatn::initTensorData("H",hadamard);
  assert(initialized);
 }

 {//Open a new scope:
  //Build a tensor network from the quantum circuit:
  TensorNetwork circuit("QuantumCircuit");
  unsigned int tensorCounter = 1;

  // Qubit tensors:
  for (unsigned int i = 0; i < nbQubits; ++i) {
   const bool appended = circuit.appendTensor(tensorCounter, exatn::getTensor("Q" + std::to_string(i)),{});
   assert(appended);
   ++tensorCounter;
  }

  // Copy the qubit reg tensor to fully-close the entire network
  TensorNetwork qubitReg(circuit);
  qubitReg.rename("QubitKet");

  // Hadamard tensors:
  for (unsigned int i = 0; i < nbQubits; ++i) {
   const bool appended = circuit.appendTensorGate(tensorCounter,exatn::getTensor("H"),{i});
   assert(appended);
   ++tensorCounter;
  }

  circuit.printIt(); //debug

  //Contract the circuit tensor network with its conjugate:
  TensorNetwork inverse(circuit);
  inverse.rename("InverseCircuit");

  for (unsigned int i = 0; i < nbQubits; ++i) {
   const bool appended = inverse.appendTensorGate(tensorCounter,exatn::getTensor("H"),{nbQubits - i - 1}, true);
   assert(appended);
   ++tensorCounter;
  }

  const bool collapsed = inverse.collapseIsometries();
  assert(collapsed);

  inverse.printIt(); //debug

  {// Closing the tensor network with the bra
   auto bra = qubitReg;
   bra.conjugate();
   bra.rename("QubitBra");
   std::vector<std::pair<unsigned int, unsigned int>> pairings;
   for (unsigned int i = 0; i < nbQubits; ++i) {
    pairings.emplace_back(std::make_pair(i, i));
   }
   inverse.appendTensorNetwork(std::move(bra), pairings);
  }

  inverse.printIt(); //debug

  {
   const bool rankEqualZero = (inverse.getRank() == 0);
   assert(rankEqualZero);
  }

  //Evaluate the quantum circuit expressed as a tensor network:
  // NOTE: We evaluate the *inverse* tensor which should be fully-closed.
  const bool evaluated = exatn::evaluateSync(inverse);
  assert(evaluated);

  //Synchronize:
  exatn::sync();

  auto talsh_tensor = exatn::getLocalTensor(inverse.getTensor(0)->getName());
  const std::complex<double>* body_ptr;
  if (talsh_tensor->getDataAccessHostConst(&body_ptr)) {
   std::cout << "Fina result is " << *body_ptr << "\n";
  }
 }

 //Destroy all tensors:
 {
  const bool destroyed = exatn::destroyTensor("H");
  assert(destroyed);
 }

 for (unsigned int i = 0; i < nbQubits; ++i) {
  const bool destroyed = exatn::destroyTensor("Q" + std::to_string(i));
  assert(destroyed);
 }

 //Synchronize:
 exatn::sync();
 //Grab a coffee!
}
#endif

#ifdef EXATN_TEST10
TEST(NumServerTester, Sycamore8NumServer)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorNetwork;
 using exatn::TensorElementType;

 //exatn::resetLoggingLevel(1,2); //debug

 const unsigned int num_qubits = 53;
 const unsigned int num_gates = 172; //total number of gates is 172
 std::vector<std::pair<unsigned int, unsigned int>> sycamore_8_cnot
 {
 {1,4},{3,7},{5,9},{6,13},{8,15},{10,17},{12,21},{14,23},{16,25},{18,27},{20,30},
 {22,32},{24,34},{26,36},{29,37},{31,39},{33,41},{35,43},{38,44},{40,46},{42,48},
 {45,49},{47,51},{50,52},{0,3},{2,6},{4,8},{7,14},{9,16},{11,20},{13,22},{15,24},
 {17,26},{19,29},{21,31},{23,33},{25,35},{30,38},{32,40},{34,42},{39,45},{41,47},
 {46,50},{0,1},{2,3},{4,5},{7,8},{9,10},{11,12},{13,14},{15,16},{17,18},{19,20},
 {21,22},{23,24},{25,26},{28,29},{30,31},{32,33},{34,35},{37,38},{39,40},{41,42},
 {44,45},{46,47},{49,50},{3,4},{6,7},{8,9},{12,13},{14,15},{16,17},{20,21},{22,23},
 {24,25},{26,27},{29,30},{31,32},{33,34},{35,36},{38,39},{40,41},{42,43},{45,46},
 {47,48},{50,51},{0,1},{2,3},{4,5},{7,8},{9,10},{11,12},{13,14},{15,16},{17,18},
 {19,20},{21,22},{23,24},{25,26},{28,29},{30,31},{32,33},{34,35},{37,38},{39,40},
 {41,42},{44,45},{46,47},{49,50},{3,4},{6,7},{8,9},{12,13},{14,15},{16,17},{20,21},
 {22,23},{24,25},{26,27},{29,30},{31,32},{33,34},{35,36},{38,39},{40,41},{42,43},
 {45,46},{47,48},{50,51},{1,4},{3,7},{5,9},{6,13},{8,15},{10,17},{12,21},{14,23},
 {16,25},{18,27},{20,30},{22,32},{24,34},{26,36},{29,37},{31,39},{33,41},{35,43},
 {38,44},{40,46},{42,48},{45,49},{47,51},{50,52},{0,3},{2,6},{4,8},{7,14},{9,16},
 {11,20},{13,22},{15,24},{17,26},{19,29},{21,31},{23,33},{25,35},{30,38},{32,40},
 {34,42},{39,45},{41,47},{46,50}
 };
 assert(num_gates <= sycamore_8_cnot.size());

 std::cout << "Building the circuit ... " << std::flush;

 TensorNetwork circuit("Sycamore8_CNOT");
 unsigned int tensor_counter = 0;

 //Left qubit tensors:
 unsigned int first_q_tensor = tensor_counter + 1;
 for(unsigned int i = 0; i < num_qubits; ++i){
  bool success = circuit.appendTensor(++tensor_counter,
                                      std::make_shared<Tensor>("Q"+std::to_string(i),TensorShape{2}),
                                      {});
  assert(success);
 }
 unsigned int last_q_tensor = tensor_counter;

 //CNOT gates:
 auto cnot = std::make_shared<Tensor>("CNOT",TensorShape{2,2,2,2});
 for(unsigned int i = 0; i < num_gates; ++i){
  bool success = circuit.appendTensorGate(++tensor_counter,
                                          cnot,
                                          {sycamore_8_cnot[i].first,sycamore_8_cnot[i].second});
  assert(success);
 }

 //Right qubit tensors:
 unsigned int first_p_tensor = tensor_counter + 1;
 for(unsigned int i = 0; i < num_qubits; ++i){
  bool success = circuit.appendTensor(++tensor_counter,
                                      std::make_shared<Tensor>("P"+std::to_string(i),TensorShape{2}),
                                      {{0,0}});
  assert(success);
 }
 unsigned int last_p_tensor = tensor_counter;
 std::cout << "Done\n" << std::flush;

 std::cout << "Simplifying the circuit ... " << std::flush;
 //Merge qubit tensors into adjacent CNOTs:
 for(unsigned int i = first_p_tensor; i <= last_p_tensor; ++i){
  const auto & tensor_legs = *(circuit.getTensorConnections(i));
  const auto other_tensor_id = tensor_legs[0].getTensorId();
  bool success = circuit.mergeTensors(other_tensor_id,i,++tensor_counter);
  assert(success);
 }
 for(unsigned int i = first_q_tensor; i <= last_q_tensor; ++i){
  const auto & tensor_legs = *(circuit.getTensorConnections(i));
  const auto other_tensor_id = tensor_legs[0].getTensorId();
  bool success = circuit.mergeTensors(other_tensor_id,i,++tensor_counter);
  assert(success);
 }
 std::cout << "Done\n" << std::flush;

 circuit.printIt(); //debug

 //Generate the list of tensor operations for the circuit:
 std::cout << "Generating the list of tensor operations for the circuit ... " << std::flush;
 auto & operations = circuit.getOperationList("metis",true);
 std::cout << "Done\n" << std::flush;
 unsigned int max_rank = 0;
 std::cout << "Total FMA flop count = " << circuit.getFMAFlops()
           << ": Max intermdediate presence volume = " << circuit.getMaxIntermediatePresenceVolume()
           << ": Max intermdediate volume = " << circuit.getMaxIntermediateVolume(&max_rank)
           << ": Max intermdediate rank = " << max_rank << std::endl;

 std::cout << "Splitting some internal indices to reduce the size of intermediates ... " << std::flush;
 circuit.splitIndices(static_cast<std::size_t>(circuit.getMaxIntermediateVolume()/16.0));
 std::cout << "Done\n" << std::flush;
 circuit.printSplitIndexInfo();

 std::size_t num_parts = 2;
 double imbalance = 1.001;
 std::size_t edge_cut = 0, num_cross_edges = 0;
 std::vector<std::pair<std::size_t,std::vector<std::size_t>>> parts;
 bool success = circuit.partition(num_parts,imbalance,parts,&edge_cut,&num_cross_edges); assert(success);
 std::cout << "Partitioned tensor network into " << num_parts
           << " parts with tolerated weight imbalance " << imbalance
           << " achieving edge cut of " << edge_cut
           << " with total cross edges = " << num_cross_edges << ":\n" << std::flush;
 std::size_t total_weight = 0;
 std::size_t total_vertices = 0;
 for(std::size_t i = 0; i < parts.size(); ++i){
  std::cout << "Partition " << i << " of size " << parts[i].second.size()
            << " with weight " << parts[i].first << ":\n";
  for(const auto & vertex: parts[i].second) std::cout << " " << vertex;
  total_weight += parts[i].first;
  total_vertices += parts[i].second.size();
  std::cout << std::endl;
 }
 std::cout << "Total weight of vertices in all partitions = " << total_weight << std::endl;
 std::cout << "Total number of vertices in all partitions = " << total_vertices << std::endl;
}
#endif

#ifdef EXATN_TEST11
TEST(NumServerTester, Sycamore12NumServer)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorNetwork;
 using exatn::TensorElementType;

 //exatn::resetLoggingLevel(1,2); //debug

 const unsigned int num_qubits = 53;
 const unsigned int num_gates = 258; //total number of gates is 258
 std::vector<std::pair<unsigned int, unsigned int>> sycamore_12_cnot
 {
  {1,4},{3,7},{5,9},{6,13},{8,15},{10,17},{12,21},{14,23},{16,25},{18,27},
  {20,30},{22,32},{24,34},{26,36},{29,37},{31,39},{33,41},{35,43},{38,44},
  {40,46},{42,48},{45,49},{47,51},{50,52},{0,3},{2,6},{4,8},{7,14},{9,16},
  {11,20},{13,22},{15,24},{17,26},{19,29},{21,31},{23,33},{25,35},{30,38},
  {32,40},{34,42},{39,45},{41,47},{46,50},{0,1},{2,3},{4,5},{7,8},{9,10},
  {11,12},{13,14},{15,16},{17,18},{19,20},{21,22},{23,24},{25,26},{28,29},
  {30,31},{32,33},{34,35},{37,38},{39,40},{41,42},{44,45},{46,47},{49,50},
  {3,4},{6,7},{8,9},{12,13},{14,15},{16,17},{20,21},{22,23},{24,25},{26,27},
  {29,30},{31,32},{33,34},{35,36},{38,39},{40,41},{42,43},{45,46},{47,48},
  {50,51},{0,1},{2,3},{4,5},{7,8},{9,10},{11,12},{13,14},{15,16},{17,18},
  {19,20},{21,22},{23,24},{25,26},{28,29},{30,31},{32,33},{34,35},{37,38},
  {39,40},{41,42},{44,45},{46,47},{49,50},{3,4},{6,7},{8,9},{12,13},{14,15},
  {16,17},{20,21},{22,23},{24,25},{26,27},{29,30},{31,32},{33,34},{35,36},
  {38,39},{40,41},{42,43},{45,46},{47,48},{50,51},{1,4},{3,7},{5,9},{6,13},
  {8,15},{10,17},{12,21},{14,23},{16,25},{18,27},{20,30},{22,32},{24,34},
  {26,36},{29,37},{31,39},{33,41},{35,43},{38,44},{40,46},{42,48},{45,49},
  {47,51},{50,52},{0,3},{2,6},{4,8},{7,14},{9,16},{11,20},{13,22},{15,24},
  {17,26},{19,29},{21,31},{23,33},{25,35},{30,38},{32,40},{34,42},{39,45},
  {41,47},{46,50},{1,4},{3,7},{5,9},{6,13},{8,15},{10,17},{12,21},{14,23},
  {16,25},{18,27},{20,30},{22,32},{24,34},{26,36},{29,37},{31,39},{33,41},
  {35,43},{38,44},{40,46},{42,48},{45,49},{47,51},{50,52},{0,3},{2,6},{4,8},
  {7,14},{9,16},{11,20},{13,22},{15,24},{17,26},{19,29},{21,31},{23,33},
  {25,35},{30,38},{32,40},{34,42},{39,45},{41,47},{46,50},{0,1},{2,3},{4,5},
  {7,8},{9,10},{11,12},{13,14},{15,16},{17,18},{19,20},{21,22},{23,24},
  {25,26},{28,29},{30,31},{32,33},{34,35},{37,38},{39,40},{41,42},{44,45},
  {46,47},{49,50},{3,4},{6,7},{8,9},{12,13},{14,15},{16,17},{20,21},{22,23},
  {24,25},{26,27},{29,30},{31,32},{33,34},{35,36},{38,39},{40,41},{42,43},
  {45,46},{47,48},{50,51}
 };
 assert(num_gates <= sycamore_12_cnot.size());

 std::cout << "Building the circuit ... " << std::flush;

 TensorNetwork circuit("Sycamore12_CNOT");
 unsigned int tensor_counter = 0;

 //Left qubit tensors:
 unsigned int first_q_tensor = tensor_counter + 1;
 for(unsigned int i = 0; i < num_qubits; ++i){
  bool success = circuit.appendTensor(++tensor_counter,
                                      std::make_shared<Tensor>("Q"+std::to_string(i),TensorShape{2}),
                                      {});
  assert(success);
 }
 unsigned int last_q_tensor = tensor_counter;

 //CNOT gates:
 auto cnot = std::make_shared<Tensor>("CNOT",TensorShape{2,2,2,2});
 for(unsigned int i = 0; i < num_gates; ++i){
  bool success = circuit.appendTensorGate(++tensor_counter,
                                          cnot,
                                          {sycamore_12_cnot[i].first,sycamore_12_cnot[i].second});
  assert(success);
 }

 //Right qubit tensors:
 unsigned int first_p_tensor = tensor_counter + 1;
 for(unsigned int i = 0; i < num_qubits; ++i){
  bool success = circuit.appendTensor(++tensor_counter,
                                      std::make_shared<Tensor>("P"+std::to_string(i),TensorShape{2}),
                                      {{0,0}});
  assert(success);
 }
 unsigned int last_p_tensor = tensor_counter;
 std::cout << "Done\n" << std::flush;

 std::cout << "Simplifying the circuit ... " << std::flush;
 //Merge qubit tensors into adjacent CNOTs:
 for(unsigned int i = first_p_tensor; i <= last_p_tensor; ++i){
  const auto & tensor_legs = *(circuit.getTensorConnections(i));
  const auto other_tensor_id = tensor_legs[0].getTensorId();
  bool success = circuit.mergeTensors(other_tensor_id,i,++tensor_counter);
  assert(success);
 }
 for(unsigned int i = first_q_tensor; i <= last_q_tensor; ++i){
  const auto & tensor_legs = *(circuit.getTensorConnections(i));
  const auto other_tensor_id = tensor_legs[0].getTensorId();
  bool success = circuit.mergeTensors(other_tensor_id,i,++tensor_counter);
  assert(success);
 }
 std::cout << "Done\n" << std::flush;

 //Decompose all higher-than-rank-3 tensors:
 //circuit.decomposeTensors(); //optional
 circuit.printIt(); //debug

 //Generate the list of tensor operations for the circuit:
 std::cout << "Generating the list of tensor operations for the circuit ... " << std::flush;
 auto & operations = circuit.getOperationList("metis",true);
 std::cout << "Done\n" << std::flush;
 unsigned int max_rank = 0;
 std::cout << "Total FMA flop count = " << circuit.getFMAFlops()
           << ": Max intermdediate presence volume = " << circuit.getMaxIntermediatePresenceVolume()
           << ": Max intermdediate volume = " << circuit.getMaxIntermediateVolume(&max_rank)
           << ": Max intermdediate rank = " << max_rank << std::endl;

 std::cout << "Splitting some internal indices to reduce the size of intermediates ... " << std::flush;
 circuit.splitIndices(static_cast<std::size_t>(circuit.getMaxIntermediateVolume()/16.0));
 std::cout << "Done\n" << std::flush;
 circuit.printSplitIndexInfo();
}
#endif

#ifdef EXATN_TEST12
TEST(NumServerTester, rcsNumServer)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorNetwork;
 using exatn::TensorElementType;

 //exatn::resetLoggingLevel(1,2); //debug

 //Configuration:
 const int NB_QUBITS = 52;
 const int NB_LAYERS = 12;
 //Define the initial qubit state vector:
 std::vector<std::complex<double>> qzero {
  {1.0,0.0}, {0.0,0.0}
 };
 //Define quantum gates:
 std::vector<std::complex<double>> hadamard {
  {1.0/sqrt(2.0),0.0}, {1.0/sqrt(2.0),0.0},
  {1.0/sqrt(2.0),0.0}, {-1.0/sqrt(2.0),0.0}
 };
 std::vector<std::complex<double>> pauliX {
  {0.0,0.0}, {1.0,0.0},
  {1.0,0.0}, {0.0,0.0}
 };
 std::vector<std::complex<double>> pauliY {
  {0.0,0.0}, {0.0,-1.0},
  {0.0,1.0}, {0.0,0.0}
 };
 std::vector<std::complex<double>> pauliZ {
  {1.0,0.0}, {0.0,0.0},
  {0.0,0.0}, {-1.0,0.0}
 };
 std::vector<std::complex<double>> cnot {
  {1.0,0.0}, {0.0,0.0}, {0.0,0.0}, {0.0,0.0},
  {0.0,0.0}, {1.0,0.0}, {0.0,0.0}, {0.0,0.0},
  {0.0,0.0}, {0.0,0.0}, {0.0,0.0}, {1.0,0.0},
  {0.0,0.0}, {0.0,0.0}, {1.0,0.0}, {0.0,0.0}
 };
 //Create qubit tensors:
 for (unsigned int i = 0; i < NB_QUBITS; ++i) {
  const bool created = exatn::createTensor("Q" + std::to_string(i),TensorElementType::COMPLEX64,TensorShape{2});
  assert(created);
 }
 //Create gate tensors:
 {
  const bool created = exatn::createTensor("H",TensorElementType::COMPLEX64,TensorShape{2,2});
  assert(created);
  const bool registered = exatn::registerTensorIsometry("H",{0},{1});
  assert(registered);
 }
 {
  const bool created = exatn::createTensor("X",TensorElementType::COMPLEX64,TensorShape{2,2});
  assert(created);
  const bool registered = exatn::registerTensorIsometry("X",{0},{1});
  assert(registered);
 }
 {
  const bool created = exatn::createTensor("Y",TensorElementType::COMPLEX64,TensorShape{2,2});
  assert(created);
  const bool registered = exatn::registerTensorIsometry("Y",{0},{1});
  assert(registered);
 }
 {
  const bool created = exatn::createTensor("Z",TensorElementType::COMPLEX64,TensorShape{2,2});
  assert(created);
  const bool registered = exatn::registerTensorIsometry("Z",{0},{1});
  assert(registered);
 }
 {
  const bool created = exatn::createTensor("CNOT",TensorElementType::COMPLEX64,TensorShape{2,2,2,2});
  assert(created);
  const bool registered = exatn::registerTensorIsometry("CNOT",{0,1},{2,3});
  assert(registered);
 }
 //Initialize qubit tensors to zero state:
 for (unsigned int i = 0; i < NB_QUBITS; ++i) {
  const bool initialized = exatn::initTensorData("Q" + std::to_string(i),qzero);
  assert(initialized);
 }
 //Initialize necessary gate tensors:
 {
  const bool initialized = exatn::initTensorData("H",hadamard);
  assert(initialized);
 }
 {
  const bool initialized = exatn::initTensorData("X",pauliX);
  assert(initialized);
 }
 {
  const bool initialized = exatn::initTensorData("Y",pauliY);
  assert(initialized);
 }
 {
  const bool initialized = exatn::initTensorData("Z",pauliZ);
  assert(initialized);
 }
 {
  const bool initialized = exatn::initTensorData("CNOT",cnot);
  assert(initialized);
 }
 //Build a tensor network from the quantum circuit:
 TensorNetwork circuit("QuantumCircuit");
 unsigned int tensorCounter = 1;
 //Qubit tensors:
 for (unsigned int i = 0; i < NB_QUBITS; ++i) {
  const bool appended = circuit.appendTensor(tensorCounter, exatn::getTensor("Q" + std::to_string(i)),{});
  assert(appended);
  ++tensorCounter;
 }
 const std::vector<std::string> GATE_SET { "H", "X", "Y", "Z" };
 for (unsigned int j = 0; j < NB_LAYERS; ++j)
 {
  for (unsigned int i = 0; i < NB_QUBITS; ++i) {
   auto randIt = GATE_SET.begin();
   std::advance(randIt, std::rand() % GATE_SET.size());
   const std::string selectedGate = *randIt;
   const bool appended = circuit.appendTensorGate(tensorCounter,exatn::getTensor(selectedGate),{i});
   assert(appended);
   ++tensorCounter;
  }
  for (unsigned int i = 0; i < NB_QUBITS - 1; ++i)
  {
   const bool appended = circuit.appendTensorGate(tensorCounter, exatn::getTensor("CNOT"),{i, i + 1});
   assert(appended);
   ++tensorCounter;
  }
 }
 circuit.printIt(); //debug
 auto inverseTensorNetwork = circuit;
 inverseTensorNetwork.rename("InverseQuantumCircuit");
 inverseTensorNetwork.conjugate();
 auto combinedNetwork = circuit;
 combinedNetwork.rename("CombinedQuantumCircuit");
 //Append the conjugate tensor network to calculate the RDM of the measure:
 std::vector<std::pair<unsigned int, unsigned int>> pairings;
 const int NB_OPEN_LEGS = 4;
 for (size_t i = NB_OPEN_LEGS; i < NB_QUBITS; ++i) {
  // Connect the original tensor network with its inverse
  // but leave the measure qubit line open:
  pairings.emplace_back(std::make_pair(i, i));
 }
 combinedNetwork.appendTensorNetwork(std::move(inverseTensorNetwork), pairings);
 const bool collapsed = combinedNetwork.collapseIsometries();
 assert(collapsed);
 combinedNetwork.printIt(); //debug
 combinedNetwork.getOperationList();
 double flops = combinedNetwork.getFMAFlops();
 double intermediates_volume = combinedNetwork.getMaxIntermediatePresenceVolume();
 std::cout << "Combined circuit requires " << flops << " FMA flops and "
           << intermediates_volume * sizeof(std::complex<double>) << " bytes\n";
 //Evaluate:
 //const bool evalOk = exatn::evaluateSync(combinedNetwork);
 //assert(evalOk);
 //Destroy all tensors:
 {
  const bool destroyed = exatn::destroyTensor("H");
  assert(destroyed);
 }
 {
  const bool destroyed = exatn::destroyTensor("X");
  assert(destroyed);
 }
 {
  const bool destroyed = exatn::destroyTensor("Y");
  assert(destroyed);
 }
 {
  const bool destroyed = exatn::destroyTensor("Z");
  assert(destroyed);
 }
 {
  const bool destroyed = exatn::destroyTensor("CNOT");
  assert(destroyed);
 }
 for (unsigned int i = 0; i < NB_QUBITS; ++i) {
  const bool destroyed = exatn::destroyTensor("Q" + std::to_string(i));
  assert(destroyed);
 }
 //Synchronize:
 exatn::sync();
 //Grab a coffee!
}
#endif

#ifdef EXATN_TEST13
TEST(NumServerTester, BigMPSNumServer)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::TensorNetwork;
 using exatn::TensorOperator;
 using exatn::TensorExpansion;
 using exatn::TensorElementType;

 //exatn::resetLoggingLevel(1,2); //debug

 const int nbQubits = 32;
 const std::vector<int> qubitTensorDim(nbQubits, 2);
 const std::string ROOT_TENSOR_NAME = "Root";
 auto rootTensor = std::make_shared<Tensor>(ROOT_TENSOR_NAME, qubitTensorDim);

 auto & networkBuildFactory = *(exatn::NetworkBuildFactory::get());
 auto builder = networkBuildFactory.createNetworkBuilderShared("MPS");
 bool success = builder->setParameter("max_bond_dim", 1);
 assert(success);

 std::cout << "Building MPS tensor network ... " << std::flush;
 auto tensorNetwork = exatn::makeSharedTensorNetwork("Qubit Register", rootTensor, *builder);
 std::cout << "Done\n" << std::flush;
 tensorNetwork->printIt();

 std::cout << "Creating/Initializing MPS tensors ... " << std::flush;
 const std::vector<std::complex<double>> ZERO_TENSOR_BODY {{1.0, 0.0}, {0.0, 0.0}};
 for(auto iter = tensorNetwork->cbegin(); iter != tensorNetwork->cend(); ++iter){
  auto tensor = iter->second.getTensor();
  const auto & tensorName = tensor->getName();
  if(tensorName != ROOT_TENSOR_NAME){
   success = exatn::createTensorSync(tensor, exatn::TensorElementType::COMPLEX64);
   assert(success);
   success = exatn::initTensorDataSync(tensorName, ZERO_TENSOR_BODY);
   assert(success);
  }
 }
 std::cout << "Done\n" << std::flush;

 exatn::TensorNetwork ket(*tensorNetwork);
 ket.rename("MPSket");
 exatn::TensorNetwork bra(ket);
 bra.conjugate();
 bra.rename("MPSbra");

 std::cout << "Constructing 1-RDM contracted tensor network ... " << std::flush;
 const std::size_t qubitIdx = 12; //qubit Id of the leg that will be left open to calculate RDM
 std::vector<std::pair<unsigned int, unsigned int>> pairings;
 for(std::size_t i = 0; i < nbQubits; ++i){
  //Connect the original tensor network with its inverse but leave the measure qubit line open:
  if(i != qubitIdx) pairings.emplace_back(std::make_pair(i,i));
 }
 success = ket.appendTensorNetwork(std::move(bra), pairings);
 assert(success);
 std::cout << "Done\n" << std::flush;

 /*
 std::cout << "Collapsing isometries ... ";
 success = ket.collapseIsometries();
 assert(success);
 std::cout << "Done\n";
 */

 std::cout << "Evaluating 1-RDM ... " << std::flush;
 success = exatn::evaluateSync(ket);
 assert(success);
 std::cout << "Done\n" << std::flush;
 exatn::sync();
}
#endif

#ifdef EXATN_TEST14
TEST(NumServerTester, HamiltonianNumServer)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::TensorNetwork;
 using exatn::TensorOperator;
 using exatn::TensorExpansion;
 using exatn::TensorElementType;

 //exatn::resetLoggingLevel(1,2); //debug

 //Declare MPS tensors:
 auto q0 = std::make_shared<Tensor>("Q0",TensorShape{2,2});
 auto q1 = std::make_shared<Tensor>("Q1",TensorShape{2,2,4});
 auto q2 = std::make_shared<Tensor>("Q2",TensorShape{4,2,2});
 auto q3 = std::make_shared<Tensor>("Q3",TensorShape{2,2});

 //Declare Hamiltonian tensors:
 auto h01 = std::make_shared<Tensor>("H01",TensorShape{2,2,2,2});
 auto h12 = std::make_shared<Tensor>("H12",TensorShape{2,2,2,2});
 auto h23 = std::make_shared<Tensor>("H23",TensorShape{2,2,2,2});
 auto z0 = std::make_shared<Tensor>("Z0",TensorShape{2,2,2,2});

 //Declare the Hamiltonian operator:
 TensorOperator ham("Hamiltonian");
 bool appended = false;
 appended = ham.appendComponent(h01,{{0,0},{1,1}},{{0,2},{1,3}},{1.0,0.0}); assert(appended);
 appended = ham.appendComponent(h12,{{1,0},{2,1}},{{1,2},{2,3}},{1.0,0.0}); assert(appended);
 appended = ham.appendComponent(h23,{{2,0},{3,1}},{{2,2},{3,3}},{1.0,0.0}); assert(appended);

 //Declare the ket MPS tensor network:
 // Q0----Q1----Q2----Q3
 // |     |     |     |
 auto mps_ket = std::make_shared<TensorNetwork>("MPS",
                 "Z0(i0,i1,i2,i3)+=Q0(i0,j0)*Q1(j0,i1,j1)*Q2(j1,i2,j2)*Q3(j2,i3)",
                 std::map<std::string,std::shared_ptr<Tensor>>{
                  {"Z0",z0}, {"Q0",q0}, {"Q1",q1}, {"Q2",q2}, {"Q3",q3}});

 //Declare the ket tensor network expansion:
 // Q0----Q1----Q2----Q3
 // |     |     |     |
 TensorExpansion ket;
 appended = ket.appendComponent(mps_ket,{1.0,0.0}); assert(appended);
 ket.rename("MPSket");

 //Declare the bra tensor network expansion (conjugated ket):
 // |     |     |     |
 // Q0----Q1----Q2----Q3
 TensorExpansion bra(ket);
 bra.conjugate();
 bra.rename("MPSbra");

 //Declare the operator times ket product tensor expansion:
 // Q0----Q1----Q2----Q3    Q0----Q1----Q2----Q3    Q0----Q1----Q2----Q3
 // |     |     |     |     |     |     |     |     |     |     |     |
 // ==H01==     |     |  +  |     ==H12==     |  +  |     |     ==H23==
 // |     |     |     |     |     |     |     |     |     |     |     |
 TensorExpansion ham_ket(ket,ham);
 ham_ket.rename("HamMPSket");

 //Declare the full closed product tensor expansion (scalar):
 // Q0----Q1----Q2----Q3    Q0----Q1----Q2----Q3    Q0----Q1----Q2----Q3
 // |     |     |     |     |     |     |     |     |     |     |     |
 // ==H01==     |     |  +  |     ==H12==     |  +  |     |     ==H23==   =>  AC0()
 // |     |     |     |     |     |     |     |     |     |     |     |
 // Q0----Q1----Q2----Q3    Q0----Q1----Q2----Q3    Q0----Q1----Q2----Q3
 TensorExpansion closed_prod(ham_ket,bra);
 closed_prod.rename("MPSbraHamMPSket");
 closed_prod.printIt(); //debug

 //Declare the derivative tensor expansion with respect to tensor Q1+:
 // Q0----Q1----Q2----Q3    Q0----Q1----Q2----Q3    Q0----Q1----Q2----Q3
 // |     |     |     |     |     |     |     |     |     |     |     |
 // ==H01==     |     |  +  |     ==H12==     |  +  |     |     ==H23==
 // |     |     |     |     |     |     |     |     |     |     |     |
 // Q0--      --Q2----Q3    Q0--      --Q2----Q3    Q0--      --Q2----Q3
 TensorExpansion deriv_q1(closed_prod,"Q1",true);
 deriv_q1.rename("DerivativeQ1");
 deriv_q1.printIt(); //debug

 {//Numerical evaluation:
  //Create MPS tensors:
  bool created = false;
  created = exatn::createTensorSync(q0,TensorElementType::COMPLEX64); assert(created);
  created = exatn::createTensorSync(q1,TensorElementType::COMPLEX64); assert(created);
  created = exatn::createTensorSync(q2,TensorElementType::COMPLEX64); assert(created);
  created = exatn::createTensorSync(q3,TensorElementType::COMPLEX64); assert(created);

  //Create Hamiltonian tensors:
  created = exatn::createTensorSync(h01,TensorElementType::COMPLEX64); assert(created);
  created = exatn::createTensorSync(h12,TensorElementType::COMPLEX64); assert(created);
  created = exatn::createTensorSync(h23,TensorElementType::COMPLEX64); assert(created);

  //Create the Accumulator tensor for the closed tensor expansion:
  created = exatn::createTensorSync("AC0",TensorElementType::COMPLEX64,TensorShape{}); assert(created);
  auto accumulator0 = exatn::getTensor("AC0");

  //Create the Accumulator tensor for the derivative tensor expansion:
  created = exatn::createTensorSync("AC1",TensorElementType::COMPLEX64,q1->getShape()); assert(created);
  auto accumulator1 = exatn::getTensor("AC1");

  //Initialize all input tensors:
  bool initialized = false;
  initialized = exatn::initTensorSync("Q0",1e-2); assert(initialized);
  initialized = exatn::initTensorSync("Q1",1e-2); assert(initialized);
  initialized = exatn::initTensorSync("Q2",1e-2); assert(initialized);
  initialized = exatn::initTensorSync("Q3",1e-2); assert(initialized);
  initialized = exatn::initTensorSync("H01",1e-2); assert(initialized);
  initialized = exatn::initTensorSync("H12",1e-2); assert(initialized);
  initialized = exatn::initTensorSync("H23",1e-2); assert(initialized);
  initialized = exatn::initTensorSync("AC0",0.0); assert(initialized);
  initialized = exatn::initTensorSync("AC1",0.0); assert(initialized);

  //Evaluate the expectation value:
  bool evaluated = false;
  evaluated = exatn::evaluateSync(closed_prod,accumulator0); assert(evaluated);

  //Evaluate the derivative of the expectation value w.r.t. tensor Q1:
  evaluated = exatn::evaluateSync(deriv_q1,accumulator1); assert(evaluated);

  //Retrieve the expectation values:
  for(auto component = closed_prod.begin(); component != closed_prod.end(); ++component){
   auto talsh_tensor = exatn::getLocalTensor(component->network_->getTensor(0)->getName());
   const std::complex<double> * body_ptr;
   auto access_granted = talsh_tensor->getDataAccessHostConst(&body_ptr); assert(access_granted);
   std::cout << "Component " << component->network_->getTensor(0)->getName() << " expectation value = "
             << *body_ptr << " VS correct value of " << 16.384*(1e-15) << std::endl;
   body_ptr = nullptr;
  }
  auto talsh_tensor = exatn::getLocalTensor("AC0"); //accumulator for the whole tensor expansion
  const std::complex<double> * body_ptr;
  auto access_granted = talsh_tensor->getDataAccessHostConst(&body_ptr); assert(access_granted);
  std::cout << "AC0 expectation value = " << *body_ptr << " VS correct value of " << 3*16.384*(1e-15) << std::endl;
  body_ptr = nullptr;

  //Destroy all tensors:
  bool destroyed = false;
  destroyed = exatn::destroyTensorSync("AC1"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("AC0"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("H23"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("H12"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("H01"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("Q3"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("Q2"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("Q1"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("Q0"); assert(destroyed);

  //Synchronize:
  exatn::sync();
 }
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST15
TEST(NumServerTester, EigenNumServer)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::TensorNetwork;
 using exatn::TensorOperator;
 using exatn::TensorExpansion;
 using exatn::TensorElementType;

 //exatn::resetLoggingLevel(1,2); //debug

 //Define Ising Hamiltonian constants:
 constexpr std::complex<double> ZERO{0.0,0.0};
 constexpr std::complex<double> HAMT{-1.0,0.0};
 constexpr std::complex<double> HAMU{-2.0,0.0};

 //Declare Ising Hamiltonian tensors:
 auto t01 = std::make_shared<Tensor>("T01",TensorShape{2,2,2,2});
 auto t12 = std::make_shared<Tensor>("T12",TensorShape{2,2,2,2});
 auto t23 = std::make_shared<Tensor>("T23",TensorShape{2,2,2,2});
 auto u00 = std::make_shared<Tensor>("U00",TensorShape{2,2});
 auto u11 = std::make_shared<Tensor>("U11",TensorShape{2,2});
 auto u22 = std::make_shared<Tensor>("U22",TensorShape{2,2});
 auto u33 = std::make_shared<Tensor>("U33",TensorShape{2,2});

 //Define Ising Hamiltonian tensor elements:
 std::vector<std::complex<double>> hamt { //Sigma_Z_i X Sigma_Z_(i+1)
  HAMT,  ZERO,  ZERO,  ZERO,
  ZERO, -HAMT,  ZERO,  ZERO,
  ZERO,  ZERO, -HAMT,  ZERO,
  ZERO,  ZERO,  ZERO,  HAMT
 };
 std::vector<std::complex<double>> hamu { //Sigma_X_i
  ZERO,  HAMU,
  HAMU,  ZERO
 };

 //Declare the Ising Hamiltonian operator:
 TensorOperator ham("Hamiltonian");
 auto appended = false;
 appended = ham.appendComponent(t01,{{0,0},{1,1}},{{0,2},{1,3}},{1.0,0.0}); assert(appended);
 appended = ham.appendComponent(t12,{{1,0},{2,1}},{{1,2},{2,3}},{1.0,0.0}); assert(appended);
 appended = ham.appendComponent(t23,{{2,0},{3,1}},{{2,2},{3,3}},{1.0,0.0}); assert(appended);
 appended = ham.appendComponent(u00,{{0,0}},{{0,1}},{1.0,0.0}); assert(appended);
 appended = ham.appendComponent(u11,{{1,0}},{{1,1}},{1.0,0.0}); assert(appended);
 appended = ham.appendComponent(u22,{{2,0}},{{2,1}},{1.0,0.0}); assert(appended);
 appended = ham.appendComponent(u33,{{3,0}},{{3,1}},{1.0,0.0}); assert(appended);

 {//Numerical evaluation:
  //Create Hamiltonian tensors:
  auto created = false;
  created = exatn::createTensorSync(t01,TensorElementType::COMPLEX64); assert(created);
  created = exatn::createTensorSync(t12,TensorElementType::COMPLEX64); assert(created);
  created = exatn::createTensorSync(t23,TensorElementType::COMPLEX64); assert(created);
  created = exatn::createTensorSync(u00,TensorElementType::COMPLEX64); assert(created);
  created = exatn::createTensorSync(u11,TensorElementType::COMPLEX64); assert(created);
  created = exatn::createTensorSync(u22,TensorElementType::COMPLEX64); assert(created);
  created = exatn::createTensorSync(u33,TensorElementType::COMPLEX64); assert(created);

  //Initialize Hamiltonian tensors:
  auto initialized = false;
  initialized = exatn::initTensorDataSync("T01",hamt); assert(initialized);
  initialized = exatn::initTensorDataSync("T12",hamt); assert(initialized);
  initialized = exatn::initTensorDataSync("T23",hamt); assert(initialized);
  initialized = exatn::initTensorDataSync("U00",hamu); assert(initialized);
  initialized = exatn::initTensorDataSync("U11",hamu); assert(initialized);
  initialized = exatn::initTensorDataSync("U22",hamu); assert(initialized);
  initialized = exatn::initTensorDataSync("U33",hamu); assert(initialized);

  //`Finish

  //Destroy all tensors:
  auto destroyed = false;
  destroyed = exatn::destroyTensorSync("U33"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("U22"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("U11"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("U00"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("T23"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("T12"); assert(destroyed);
  destroyed = exatn::destroyTensorSync("T01"); assert(destroyed);

  //Synchronize:
  exatn::sync();
 }

}
#endif

#ifdef EXATN_TEST16
TEST(NumServerTester, MPSBuilderNumServer)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::TensorNetwork;
 using exatn::TensorOperator;
 using exatn::TensorExpansion;
 using exatn::TensorElementType;

 //exatn::resetLoggingLevel(1,2); //debug

 auto & networkBuildFactory = *(exatn::NetworkBuildFactory::get());
 auto builder = networkBuildFactory.createNetworkBuilderShared("MPS");
 bool success = builder->setParameter("max_bond_dim",1);
 assert(success);
 const std::string ROOT_TENSOR_NAME = "Root";
 success = exatn::createTensorSync(ROOT_TENSOR_NAME, TensorElementType::COMPLEX64, TensorShape{2,2,2,2});
 assert(success);
 auto rootTensor = exatn::getTensor(ROOT_TENSOR_NAME);
 success = exatn::initTensorSync(ROOT_TENSOR_NAME, 0.0);
 assert(success);
 auto tensorNetwork = exatn::makeSharedTensorNetwork("Qubit Register", rootTensor, *builder);
 tensorNetwork->printIt();
 const std::vector<std::complex<double>> ZERO_TENSOR_BODY {{1.0, 0.0}, {0.0, 0.0}};
 for(auto iter = tensorNetwork->cbegin(); iter != tensorNetwork->cend(); ++iter){
  auto tensor = iter->second.getTensor();
  const auto & tensorName = tensor->getName();
  if(tensorName != ROOT_TENSOR_NAME){
   success = exatn::createTensorSync(tensor, exatn::TensorElementType::COMPLEX64);
   assert(success);
   success = exatn::initTensorDataSync(tensorName, ZERO_TENSOR_BODY);
   assert(success);
  }
 }
 success = exatn::evaluateSync(*tensorNetwork);
 assert(success);
 exatn::sync();

}
#endif

#ifdef EXATN_TEST17
TEST(NumServerTester, TestSVD)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::TensorNetwork;
 using exatn::TensorOperator;
 using exatn::TensorExpansion;
 using exatn::TensorElementType;

 bool success = true;

 success = exatn::createTensor("D",TensorElementType::REAL64,TensorShape{2,2,2,2,2}); assert(success);
 success = exatn::createTensor("L",TensorElementType::REAL64,TensorShape{2,2,2,2}); assert(success);
 success = exatn::createTensor("R",TensorElementType::REAL64,TensorShape{2,2,2,2,2}); assert(success);
 success = exatn::createTensor("S",TensorElementType::REAL64,TensorShape{2,2}); assert(success);

 success = exatn::initTensorRndSync("D"); assert(success);
 success = exatn::initTensorSync("L",0.0); assert(success);
 success = exatn::initTensorSync("R",0.0); assert(success);
 success = exatn::initTensorSync("S",0.0); assert(success);

 exatn::sync();

 success = exatn::decomposeTensorSVDSync("D(a,b,c,d,e)=L(c,i,e,j)*S(i,j)*R(b,j,a,i,d)"); assert(success);

 success = exatn::destroyTensor("S"); assert(success);
 success = exatn::destroyTensor("R"); assert(success);
 success = exatn::destroyTensor("L"); assert(success);
 success = exatn::destroyTensor("D"); assert(success);

 exatn::sync();
}
#endif

#ifdef EXATN_TEST18
TEST(NumServerTester, ParserExaTN)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::TensorNetwork;
 using exatn::TensorOperator;
 using exatn::TensorExpansion;
 using exatn::TensorElementType;

 //Register a user-defined tensor method (simply performs random initialization):
 exatn::registerTensorMethod("ComputeTwoBodyHamiltonian",
                             std::shared_ptr<exatn::TensorMethod>{new exatn::numerics::FunctorInitRnd()});

 //Externally provided std::vector with user data used to init T2 (simply set to a specific value):
 const std::size_t t2_tensor_volume = (84-42+1)*(84-42+1)*(127-0+1)*(127-0+1); //see T2 shape below
 std::vector<std::complex<float>> data_vector(t2_tensor_volume,std::complex<float>{-1e-3,1e-4});

 //ExaTN code generated from a sample TAProL source (see the exatn::parser test):
 exatn::openScope("main");
 auto _my_space = exatn::createVectorSpace("my_space",(255 - 0 + 1));
 auto _your_space = exatn::createVectorSpace("your_space",(511 - 0 + 1));
 auto _s0 = exatn::createSubspace("s0","my_space",std::pair<exatn::DimOffset,exatn::DimOffset>{0,127});
 auto _s0_space = _my_space;
 auto _s1 = exatn::createSubspace("s1","my_space",std::pair<exatn::DimOffset,exatn::DimOffset>{128,255});
 auto _s1_space = _my_space;
 auto _r0 = exatn::createSubspace("r0","your_space",std::pair<exatn::DimOffset,exatn::DimOffset>{42,84});
 auto _r0_space = _your_space;
 auto _r1 = exatn::createSubspace("r1","your_space",std::pair<exatn::DimOffset,exatn::DimOffset>{484,511});
 auto _r1_space = _your_space;
 auto _i = _s0;
 auto _i_space = _s0_space;
 auto _j = _s0;
 auto _j_space = _s0_space;
 auto _k = _s0;
 auto _k_space = _s0_space;
 auto _l = _s0;
 auto _l_space = _s0_space;
 auto _a = _r0;
 auto _a_space = _r0_space;
 auto _b = _r0;
 auto _b_space = _r0_space;
 auto _c = _r0;
 auto _c_space = _r0_space;
 auto _d = _r0;
 auto _d_space = _r0_space;
 exatn::createTensor("H2",TensorSignature{{_a_space,_a},{_i_space,_i},{_b_space,_b},{_j_space,_j}},exatn::TensorElementType::COMPLEX32);
 exatn::initTensor("H2",std::complex<float>(0.0,0.0));
 exatn::transformTensor("H2","ComputeTwoBodyHamiltonian");
 exatn::createTensor("T2",TensorSignature{{_a_space,_a},{_b_space,_b},{_i_space,_i},{_j_space,_j}},exatn::TensorElementType::COMPLEX32);
 exatn::initTensorData("T2",data_vector);
 exatn::createTensor("Z2",TensorSignature{{_a_space,_a},{_b_space,_b},{_i_space,_i},{_j_space,_j}},exatn::TensorElementType::COMPLEX32);
 exatn::initTensor("Z2",std::complex<float>(0.0,0.0));
 exatn::contractTensors("Z2(a,b,i,j)+=H2(a,k,c,i)*T2(b,c,k,j)",1.0);
 exatn::evaluateTensorNetwork("_SmokyTN","Z2(a,b,i,j)+=H2(c,k,d,l)*T2(c,d,i,j)*T2(a,b,k,l)");
 exatn::scaleTensor("Z2",0.25);
 exatn::addTensors("T2(a,b,i,j)+=Z2(a,b,i,j)",1.0);
 {
  auto talsh_t2 = exatn::getLocalTensor("T2");
  exatn::createTensor("X2",TensorSignature{},exatn::TensorElementType::COMPLEX32);
  exatn::initTensor("X2",std::complex<float>(0.0,0.0));
  exatn::contractTensors("X2()+=Z2+(a,b,i,j)*Z2(a,b,i,j)",1.0);
  double norm_x2;
  exatn::computeNorm1Sync("X2",norm_x2);
  auto talsh_x2 = exatn::getLocalTensor("X2");
 }
 exatn::destroyTensor("X2");
 exatn::destroyTensor("Z2");
 exatn::destroyTensor("T2");
 exatn::destroyTensor("H2");
 exatn::closeScope();
}
#endif

#ifdef EXATN_TEST19
TEST(NumServerTester, testGarbage) {
 using exatn::TensorShape;
 using exatn::Tensor;
 using exatn::TensorNetwork;
 using exatn::TensorElementType;

 //exatn::resetLoggingLevel(2,2); // debug

 // Define the initial qubit state vector:
 std::vector<std::complex<double>> qzero{
     {1.0, 0.0}, {0.0, 0.0}};

 // Define quantum gates:
 std::vector<std::complex<double>> hadamard{
     {1.0, 0.0}, {1.0, 0.0},
     {1.0, 0.0}, {-1.0, 0.0}};
 std::vector<std::complex<double>> cnot{
     {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
     {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
     {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0},
     {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

 // Create qubit tensors:
 auto success = true;
 const int NB_QUBITS = 25;

 for (int i = 0; i < NB_QUBITS; ++i) {
  success = exatn::createTensor("Q" + std::to_string(i), TensorElementType::COMPLEX64, TensorShape{2});
  assert(success);
 }

 // Create gate tensors:
 success = exatn::createTensor("H", TensorElementType::COMPLEX64, TensorShape{2, 2}); assert(success);
 success = exatn::registerTensorIsometry("H", {0}, {1}); assert(success);
 success = exatn::createTensor("CNOT", TensorElementType::COMPLEX64, TensorShape{2, 2, 2, 2}); assert(success);
 success = exatn::registerTensorIsometry("CNOT", {0, 1}, {2, 3}); assert(success);

 // Initialize qubit tensors to zero state:
 for (int i = 0; i < NB_QUBITS; ++i) {
  success = exatn::initTensorData("Q" + std::to_string(i), qzero); assert(success);
 }

 // Initialize necessary gate tensors:
 success = exatn::initTensorData("H", hadamard); assert(success);
 success = exatn::initTensorData("CNOT", cnot); assert(success);

 { // Open a new scope:
  // Build a tensor network from the quantum circuit:
  TensorNetwork circuit("QuantumCircuit");
  int tensorCounter = 1;
  for (int i = 0; i < NB_QUBITS; ++i) {
   success = circuit.appendTensor(tensorCounter++, exatn::getTensor("Q" + std::to_string(i)), {});
   assert(success);
  }
  for (unsigned int i = 0; i < NB_QUBITS; ++i) {
   success = circuit.appendTensorGate(tensorCounter++, exatn::getTensor("H"), {i});
   assert(success);
  }

  success = circuit.appendTensorGate(tensorCounter++, exatn::getTensor("CNOT"), {1, 2});
  assert(success);

  const auto testFunc = [](const TensorNetwork &in_tensorNet) {
   auto tenNetCopy = in_tensorNet;
   int tensorIdCounter = 1;
   TensorNetwork appendTenNet("some_net");
   std::vector<std::pair<unsigned int, unsigned int>> pairings;
   for (int i = 0; i < NB_QUBITS; ++i) {
    const std::string braQubitName = "QB" + std::to_string(i);
    const bool created = exatn::createTensor(braQubitName, TensorElementType::COMPLEX64, TensorShape{2, 2});
    assert(created);
    const bool initialized = exatn::initTensorData(braQubitName,
               std::vector<std::complex<double>>{{1.0, 0.0}, {0.0, 0.0},
                                                 {0.0, 0.0}, {1.0, 0.0}});
    assert(initialized);
    pairings.emplace_back(std::make_pair(i, 2 * i));

    appendTenNet.appendTensor(tensorIdCounter, exatn::getTensor(braQubitName),
                              std::vector<std::pair<unsigned int, unsigned int>>{});
    tensorIdCounter++;
   }

   appendTenNet.conjugate();
   tenNetCopy.appendTensorNetwork(std::move(appendTenNet), pairings);
   tenNetCopy.collapseIsometries();
   // Evaluate the quantum circuit expressed as a tensor network:
   //in_tensorNet.printIt(); // debug
   //tenNetCopy.printIt(); // debug
   auto evaluated = exatn::evaluateSync(tenNetCopy); assert(evaluated);

   {
    std::vector<std::complex<double>> talshVec;
    auto talsh_tensor = exatn::getLocalTensor(tenNetCopy.getTensor(0)->getName()); assert(talsh_tensor);
    const std::complex<double> *body_ptr;
    if (talsh_tensor->getDataAccessHostConst(&body_ptr)) {
        talshVec.assign(body_ptr, body_ptr + talsh_tensor->getVolume());
    }
   }

   //bool success = exatn::destroyTensor(tenNetCopy.getTensor(0)->getName()); assert(success);
   for (int i = 0; i < NB_QUBITS; ++i) {
    const bool destroyed = exatn::destroyTensor("QB" + std::to_string(i));
    assert(destroyed);
   }
  };

  for (int i = 0; i < 16; ++i) {
   //std::cout << "Run " << i << "\n"; //debug
   testFunc(circuit);
   exatn::sync();
   //exatn::numericalServer->printImplicitTensors(); //debug
  }

  // Synchronize:
  exatn::sync();
 }

 // Destroy all tensors:
 auto destroyed = false;
 destroyed = exatn::destroyTensor("CNOT"); assert(destroyed);
 destroyed = exatn::destroyTensor("H"); assert(destroyed);
 for (int i = 0; i < NB_QUBITS; ++i) {
  destroyed = exatn::destroyTensor("Q" + std::to_string(i));
  assert(destroyed);
 }

 // Synchronize:
 exatn::sync();
 // Grab a beer!
}
#endif

#ifdef EXATN_TEST20
TEST(NumServerTester, testHyper) {
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::Tensor;
 using exatn::TensorNetwork;
 using exatn::TensorElementType;

 const auto TENS_ELEM_TYPE = TensorElementType::COMPLEX64;

 const auto ltens_val = std::complex<double>{0.001,-0.0001};
 const auto rtens_val = std::complex<double>{0.002,-0.0002};

 //exatn::resetLoggingLevel(1,2); //debug

 bool success = true;

 //Open new scope:
 {
  double norm1;

  //Create tensors:
  success = exatn::createTensor("D",TENS_ELEM_TYPE,TensorShape{48,24,320}); assert(success);
  success = exatn::createTensor("L",TENS_ELEM_TYPE,TensorShape{320,48,48,24}); assert(success);
  success = exatn::createTensor("R",TENS_ELEM_TYPE,TensorShape{24,320,48,24}); assert(success);

  //Initialize tensors:
  success = exatn::initTensor("D",std::complex<double>{0.0,0.0}); assert(success);
  success = exatn::initTensor("L",ltens_val); assert(success);
  success = exatn::initTensor("R",rtens_val); assert(success);
  success = exatn::sync(); assert(success);

  //Contract tensors:
  success = exatn::contractTensorsSync("D(i,j,k)+=L(k,a,i,b)*R(b,k,a,j)",0.25); assert(success);
  auto time_start = exatn::Timer::timeInSecHR();
  success = exatn::contractTensors("D(i,j,k)+=L(k,a,i,b)*R(b,k,a,j)",0.25); assert(success);
  success = exatn::contractTensors("D(i,j,k)+=L(k,a,i,b)*R(b,k,a,j)",0.25); assert(success);
  success = exatn::contractTensors("D(i,j,k)+=L(k,a,i,b)*R(b,k,a,j)",0.25); assert(success);
  success = exatn::sync(); assert(success);
  auto duration = exatn::Timer::timeInSecHR(time_start);
  std::cout << "Average performance (GFlop/s) = " << 3.0*8.0*(48.0*24.0*320.0*48.0*24.0)/duration/1e9 << std::endl;

  //Check correctness:
  const double correct_norm1 = std::abs(std::complex<double>{48.0 * 24.0} *
   std::abs(ltens_val * rtens_val) * std::complex<double>{48.0 * 24.0 * 320.0});
  success = exatn::computeNorm1Sync("D",norm1); assert(success);
  std::cout << "Result norm = " << norm1 << " VS correct = " << correct_norm1 << std::endl;

  //Destroy tensors:
  success = exatn::destroyTensor("R"); assert(success);
  success = exatn::destroyTensor("L"); assert(success);
  success = exatn::destroyTensor("D"); assert(success);
 }

 //Synchronize:
 success = exatn::sync(); assert(success);
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST21
TEST(NumServerTester, neurIPS) {
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::Tensor;
 using exatn::TensorNetwork;
 using exatn::TensorElementType;

 const auto TENS_ELEM_TYPE = TensorElementType::COMPLEX32;

 //exatn::resetLoggingLevel(2,2); //debug

 bool success = true;

 //3:1 1D MERA:
 {
  std::cout << "Evaluating a 3:1 MERA 1D diagram: ";
  const exatn::DimExtent chi = 18;
  success = exatn::createTensor("Z",TENS_ELEM_TYPE,TensorShape{chi,chi,chi,chi}); assert(success);
  success = exatn::createTensor("A",TENS_ELEM_TYPE,TensorShape{chi,chi,chi,chi}); assert(success);
  success = exatn::createTensor("B",TENS_ELEM_TYPE,TensorShape{chi,chi,chi,chi}); assert(success);
  success = exatn::createTensor("C",TENS_ELEM_TYPE,TensorShape{chi,chi,chi,chi}); assert(success);
  success = exatn::createTensor("D",TENS_ELEM_TYPE,TensorShape{chi,chi,chi,chi}); assert(success);
  success = exatn::createTensor("E",TENS_ELEM_TYPE,TensorShape{chi,chi,chi,chi}); assert(success);
  success = exatn::createTensor("F",TENS_ELEM_TYPE,TensorShape{chi,chi,chi,chi}); assert(success);
  success = exatn::createTensor("G",TENS_ELEM_TYPE,TensorShape{chi,chi,chi,chi}); assert(success);

  success = exatn::initTensor("Z",std::complex<float>{0.0f,0.0f}); assert(success);
  success = exatn::initTensorRnd("A"); assert(success);
  success = exatn::initTensorRnd("B"); assert(success);
  success = exatn::initTensorRnd("C"); assert(success);
  success = exatn::initTensorRnd("D"); assert(success);
  success = exatn::initTensorRnd("E"); assert(success);
  success = exatn::initTensorRnd("F"); assert(success);
  success = exatn::initTensorRnd("G"); assert(success);
  success = exatn::sync(); assert(success);

  auto time_start = exatn::Timer::timeInSecHR();
  success = exatn::evaluateTensorNetwork("MERA1d","Z(z0,z1,z2,z3)+=A(a0,a1,a2,z2)*B(b0,b1,b2,z3)*C(c0,c1,a2,b0)*D(d0,d1,a1,c0)*E(e2,e3,d1,c1)*F(a0,d0,e2,z0)*G(e3,b1,b2,z1)");
  assert(success);
  success = exatn::sync(); assert(success);
  auto duration = exatn::Timer::timeInSecHR(time_start);
  std::cout << "Time (s) = " << duration << std::endl << std::flush;

  success = exatn::destroyTensor("G"); assert(success);
  success = exatn::destroyTensor("F"); assert(success);
  success = exatn::destroyTensor("E"); assert(success);
  success = exatn::destroyTensor("D"); assert(success);
  success = exatn::destroyTensor("C"); assert(success);
  success = exatn::destroyTensor("B"); assert(success);
  success = exatn::destroyTensor("A"); assert(success);
  success = exatn::destroyTensor("Z"); assert(success);
  success = exatn::sync(); assert(success);
 }

 //AIEM 2:1 TTN:
 {
  std::cout << "Evaluating an AIEM 2:1 TTN diagram: ";
  const exatn::DimExtent chi1 = 2;
  const auto chi2 = chi1*chi1;
  const auto chi3 = chi2*chi1;
  const auto chi4 = chi3*chi1;
  success = exatn::createTensor("Z",TENS_ELEM_TYPE,TensorShape{chi3,chi3,chi4}); assert(success);
  success = exatn::createTensor("A",TENS_ELEM_TYPE,TensorShape{chi4,chi4}); assert(success);
  success = exatn::createTensor("B",TENS_ELEM_TYPE,TensorShape{chi3,chi3,chi4}); assert(success);
  success = exatn::createTensor("C",TENS_ELEM_TYPE,TensorShape{chi3,chi3,chi4}); assert(success);
  success = exatn::createTensor("D",TENS_ELEM_TYPE,TensorShape{chi2,chi2,chi3}); assert(success);
  success = exatn::createTensor("E",TENS_ELEM_TYPE,TensorShape{chi2,chi2,chi3}); assert(success);
  success = exatn::createTensor("F",TENS_ELEM_TYPE,TensorShape{chi1,chi1,chi2}); assert(success);
  success = exatn::createTensor("G",TENS_ELEM_TYPE,TensorShape{chi1,chi1,chi2}); assert(success);
  success = exatn::createTensor("H",TENS_ELEM_TYPE,TensorShape{chi1,chi1,chi1,chi1}); assert(success);
  success = exatn::createTensor("I",TENS_ELEM_TYPE,TensorShape{chi1,chi1,chi2}); assert(success);
  success = exatn::createTensor("J",TENS_ELEM_TYPE,TensorShape{chi1,chi1,chi2}); assert(success);
  success = exatn::createTensor("K",TENS_ELEM_TYPE,TensorShape{chi2,chi2,chi3}); assert(success);
  success = exatn::createTensor("L",TENS_ELEM_TYPE,TensorShape{chi2,chi2,chi3}); assert(success);
  success = exatn::createTensor("M",TENS_ELEM_TYPE,TensorShape{chi3,chi3,chi4}); assert(success);
  success = exatn::createTensor("N",TENS_ELEM_TYPE,TensorShape{chi4,chi4}); assert(success);

  success = exatn::initTensor("Z",std::complex<float>{0.0f,0.0f}); assert(success);
  success = exatn::initTensorRnd("A"); assert(success);
  success = exatn::initTensorRnd("B"); assert(success);
  success = exatn::initTensorRnd("C"); assert(success);
  success = exatn::initTensorRnd("D"); assert(success);
  success = exatn::initTensorRnd("E"); assert(success);
  success = exatn::initTensorRnd("F"); assert(success);
  success = exatn::initTensorRnd("G"); assert(success);
  success = exatn::initTensorRnd("H"); assert(success);
  success = exatn::initTensorRnd("I"); assert(success);
  success = exatn::initTensorRnd("J"); assert(success);
  success = exatn::initTensorRnd("K"); assert(success);
  success = exatn::initTensorRnd("L"); assert(success);
  success = exatn::initTensorRnd("M"); assert(success);
  success = exatn::initTensorRnd("N"); assert(success);
  success = exatn::sync(); assert(success);

  auto time_start = exatn::Timer::timeInSecHR();
  success = exatn::evaluateTensorNetwork("AIEM_TTN","Z(z0,z1,z2)+=A(a0,a1)*B(b0,b1,a0)*C(c0,z1,a1)*D(d0,d1,b1)*E(e0,e1,c0)*F(f0,f1,d1)*G(g0,g1,e0)*H(h0,h1,f1,g0)*I(f0,h0,i2)*J(h1,g1,j2)*K(d0,i2,k2)*L(j2,e1,z0)*M(b0,k2,m2)*N(m2,z2)");
  assert(success);
  success = exatn::sync(); assert(success);
  auto duration = exatn::Timer::timeInSecHR(time_start);
  std::cout << "Time (s) = " << duration << std::endl << std::flush;

  success = exatn::destroyTensor("N"); assert(success);
  success = exatn::destroyTensor("M"); assert(success);
  success = exatn::destroyTensor("L"); assert(success);
  success = exatn::destroyTensor("K"); assert(success);
  success = exatn::destroyTensor("J"); assert(success);
  success = exatn::destroyTensor("I"); assert(success);
  success = exatn::destroyTensor("H"); assert(success);
  success = exatn::destroyTensor("G"); assert(success);
  success = exatn::destroyTensor("F"); assert(success);
  success = exatn::destroyTensor("E"); assert(success);
  success = exatn::destroyTensor("D"); assert(success);
  success = exatn::destroyTensor("C"); assert(success);
  success = exatn::destroyTensor("B"); assert(success);
  success = exatn::destroyTensor("A"); assert(success);
  success = exatn::destroyTensor("Z"); assert(success);
  success = exatn::sync(); assert(success);
 }

 //ML MERA:
 {
  std::cout << "Evaluating an ML MERA diagram: ";
  const exatn::DimExtent chi1 = 3;
  const auto chi2 = chi1*chi1;
  const auto chi4 = chi2*chi2;
  success = exatn::createTensor("Z",TENS_ELEM_TYPE,TensorShape{chi1,chi1,chi1}); assert(success);
  success = exatn::createTensor("A",TENS_ELEM_TYPE,TensorShape{chi1,chi1,chi1}); assert(success);
  success = exatn::createTensor("B",TENS_ELEM_TYPE,TensorShape{chi1,chi1,chi1,chi1}); assert(success);
  success = exatn::createTensor("C",TENS_ELEM_TYPE,TensorShape{chi1,chi2}); assert(success);
  success = exatn::createTensor("D",TENS_ELEM_TYPE,TensorShape{chi1,chi1,chi2}); assert(success);
  success = exatn::createTensor("E",TENS_ELEM_TYPE,TensorShape{chi1,chi2}); assert(success);
  success = exatn::createTensor("F",TENS_ELEM_TYPE,TensorShape{chi2,chi2,chi2}); assert(success);
  success = exatn::createTensor("G",TENS_ELEM_TYPE,TensorShape{chi2,chi2,chi2,chi2}); assert(success);
  success = exatn::createTensor("H",TENS_ELEM_TYPE,TensorShape{chi2,chi4}); assert(success);
  success = exatn::createTensor("I",TENS_ELEM_TYPE,TensorShape{chi2,chi2,chi4}); assert(success);
  success = exatn::createTensor("J",TENS_ELEM_TYPE,TensorShape{chi2,chi4}); assert(success);
  success = exatn::createTensor("K",TENS_ELEM_TYPE,TensorShape{chi4,chi4,chi4}); assert(success);

  success = exatn::initTensor("Z",std::complex<float>{0.0f,0.0f}); assert(success);
  success = exatn::initTensorRnd("A"); assert(success);
  success = exatn::initTensorRnd("B"); assert(success);
  success = exatn::initTensorRnd("C"); assert(success);
  success = exatn::initTensorRnd("D"); assert(success);
  success = exatn::initTensorRnd("E"); assert(success);
  success = exatn::initTensorRnd("F"); assert(success);
  success = exatn::initTensorRnd("G"); assert(success);
  success = exatn::initTensorRnd("H"); assert(success);
  success = exatn::initTensorRnd("I"); assert(success);
  success = exatn::initTensorRnd("J"); assert(success);
  success = exatn::initTensorRnd("K"); assert(success);
  success = exatn::sync(); assert(success);

  auto time_start = exatn::Timer::timeInSecHR();
  success = exatn::evaluateTensorNetwork("ML_MERA","Z(z0,z1,z2)+=A(z0,a1,a2)*B(z1,z2,b2,b3)*C(a1,c1)*D(a2,b2,d2)*E(b3,e1)*F(c1,f1,f2)*G(d2,e1,g2,g3)*H(f1,h1)*I(f2,g2,i2)*J(g3,j1)*K(h1,i2,j1)");
  assert(success);
  success = exatn::sync(); assert(success);
  auto duration = exatn::Timer::timeInSecHR(time_start);
  std::cout << "Time (s) = " << duration << std::endl << std::flush;

  success = exatn::destroyTensor("K"); assert(success);
  success = exatn::destroyTensor("J"); assert(success);
  success = exatn::destroyTensor("I"); assert(success);
  success = exatn::destroyTensor("H"); assert(success);
  success = exatn::destroyTensor("G"); assert(success);
  success = exatn::destroyTensor("F"); assert(success);
  success = exatn::destroyTensor("E"); assert(success);
  success = exatn::destroyTensor("D"); assert(success);
  success = exatn::destroyTensor("C"); assert(success);
  success = exatn::destroyTensor("B"); assert(success);
  success = exatn::destroyTensor("A"); assert(success);
  success = exatn::destroyTensor("Z"); assert(success);
  success = exatn::sync(); assert(success);
 }

 //Synchronize:
 success = exatn::sync(); assert(success);
 exatn::resetLoggingLevel(0,0);
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST22
TEST(NumServerTester, MPSNorm) {
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::Tensor;
 using exatn::TensorNetwork;
 using exatn::TensorElementType;

 const auto TENS_ELEM_TYPE = TensorElementType::COMPLEX32;

 exatn::resetLoggingLevel(1,2); //debug

 bool success = true;

 const int num_qubits = 8;
 auto output_tensor = std::make_shared<Tensor>("Z0", std::vector<unsigned int>(num_qubits, 2));

 std::cout << "Building MPS tensor network ... " << std::flush;
 auto builder = exatn::getTensorNetworkBuilder("MPS");
 success = builder->setParameter("max_bond_dim", 256); assert(success);
 TensorNetwork mps("QubitRegister", output_tensor, *builder);
 std::cout << "Done" << std::endl << std::flush;
 mps.printIt();

 std::cout << "Building MPS norm tensor network ... " << std::flush;
 TensorNetwork mps_norm(mps);
 mps_norm.rename("MPSNorm");
 success = mps.conjugate(); assert(success);
 std::vector<std::pair<unsigned int, unsigned int>> pairing(num_qubits, {0,0});
 for(int i = 0; i < num_qubits; ++i) pairing[i] = {i,i};
 mps_norm.appendTensorNetwork(std::move(mps),pairing);
 std::cout << "Done" << std::endl << std::flush;
 mps_norm.printIt();

 std::cout << "Allocating tensor storage ... " << std::flush;
 success = exatn::createTensors(mps_norm,TENS_ELEM_TYPE); assert(success);
 std::cout << "Done" << std::endl << std::flush;

 std::cout << "Initializing tensors ... " << std::flush;
 success = exatn::initTensorsRnd(mps_norm); assert(success);
 success = exatn::sync(); assert(success);
 std::cout << "Done" << std::endl << std::flush;

 std::cout << "Determining tensor contraction sequence ... " << std::flush;
 auto flops = mps_norm.determineContractionSequence("metis");
 std::cout << "Done" << std::endl << std::flush;
 exatn::printContractionSequence(mps_norm.exportContractionSequence());

 std::cout << "Evaluating tensor network ... " << std::flush;
 success = exatn::evaluate(mps_norm); assert(success);
 success = exatn::sync(); assert(success);
 std::cout << "Done" << std::endl << std::flush;

 std::cout << "Destroying tensors ... " << std::flush;
 success = exatn::destroyTensors(mps_norm); assert(success);
 std::cout << "Done" << std::endl << std::flush;

 //Synchronize:
 success = exatn::sync(); assert(success);
 exatn::resetLoggingLevel(0,0);
 //Grab a beer!
}
#endif


int main(int argc, char **argv) {

  exatn::ParamConf exatn_parameters;
  //Set the available CPU Host RAM size to be used by ExaTN:
  exatn_parameters.setParameter("host_memory_buffer_size",8L*1024L*1024L*1024L);
#ifdef MPI_ENABLED
  int thread_provided;
  int mpi_error = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_provided);
  assert(mpi_error == MPI_SUCCESS);
  assert(thread_provided == MPI_THREAD_MULTIPLE);
  exatn::initialize(exatn::MPICommProxy(MPI_COMM_WORLD),exatn_parameters,"lazy-dag-executor");
#else
  exatn::initialize(exatn_parameters,"lazy-dag-executor");
#endif

  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();

  exatn::finalize();
#ifdef MPI_ENABLED
  mpi_error = MPI_Finalize(); assert(mpi_error == MPI_SUCCESS);
#endif
  return ret;
}
