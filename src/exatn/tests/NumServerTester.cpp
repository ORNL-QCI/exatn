#include <gtest/gtest.h>

#include "exatn.hpp"
#include "quantum.hpp"
#include "talshxx.hpp"

#ifdef MPI_ENABLED
#include "mpi.h"
#endif

#include <iostream>
#include <ios>
#include <utility>
#include <numeric>
#include <chrono>
#include <thread>

#include "errors.hpp"

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
//#define EXATN_TEST12
#define EXATN_TEST13
#define EXATN_TEST14
//#define EXATN_TEST15 //buggy (parsed named spaces/subspaces)
#define EXATN_TEST16
//#define EXATN_TEST17 //MKL only (tensor hyper-contraction)
#define EXATN_TEST18
#define EXATN_TEST19
#define EXATN_TEST20
#define EXATN_TEST21
#define EXATN_TEST22
#define EXATN_TEST23
#define EXATN_TEST24
#define EXATN_TEST25*/
#define EXATN_TEST26
/*//#define EXATN_TEST27 //requires input file from source
//#define EXATN_TEST28 //requires input file from source
#define EXATN_TEST29
//#define EXATN_TEST30
//#define EXATN_TEST31 //requires input file from source
//#define EXATN_TEST32
#define EXATN_TEST33
//#define EXATN_TEST34*/


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

 const exatn::DimExtent DIM = 1024; //CPU: 1024 for low-end CPU, 2048 for high-end CPU
                                    //3072 for Maxwell, 4096 for Pascal and Volta
 const auto TENS_ELEM_TYPE = TensorElementType::REAL32;

 //exatn::resetLoggingLevel(1,2); //debug
 //exatn::resetExecutionSerialization(true,true); //debug
 //exatn::activateFastMath(); //fast math (mixed-precision)

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

 bool success = true;
 bool root = (exatn::getProcessRank() == 0);
 if(root) std::cout << "Contractions of rank-2 tensors:" << std::endl;

 //Create tensors:
 if(root) std::cout << " Creating all tensors ... ";
 success = exatn::createTensor("A",TENS_ELEM_TYPE,TensorShape{DIM,DIM}); assert(success);
 success = exatn::createTensor("B",TENS_ELEM_TYPE,TensorShape{DIM,DIM}); assert(success);
 success = exatn::createTensor("C",TENS_ELEM_TYPE,TensorShape{DIM,DIM}); assert(success);
 success = exatn::createTensor("D",TENS_ELEM_TYPE,TensorShape{DIM,DIM}); assert(success);
 success = exatn::createTensor("E",TENS_ELEM_TYPE,TensorShape{DIM,DIM}); assert(success);
 success = exatn::createTensor("F",TENS_ELEM_TYPE,TensorShape{DIM,DIM}); assert(success);
 success = exatn::createTensor("G",TENS_ELEM_TYPE,TensorShape{DIM,DIM}); assert(success);
 success = exatn::createTensor("H",TENS_ELEM_TYPE,TensorShape{DIM,DIM}); assert(success);
 success = exatn::createTensor("I",TENS_ELEM_TYPE,TensorShape{DIM,DIM}); assert(success);
 if(root) std::cout << "Done\n";

 //Initialize tensors:
 if(root) std::cout << " Initializing all tensors ... ";
 success = exatn::initTensor("A",1e-4); assert(success);
 success = exatn::initTensor("B",1e-3); assert(success);
 success = exatn::initTensor("C",0.0); assert(success);
 success = exatn::initTensor("D",1e-4); assert(success);
 success = exatn::initTensor("E",1e-3); assert(success);
 success = exatn::initTensor("F",0.0); assert(success);
 success = exatn::initTensor("G",1e-4); assert(success);
 success = exatn::initTensor("H",1e-3); assert(success);
 success = exatn::initTensor("I",0.0); assert(success);
 if(root) std::cout << "Done\n";

 std::this_thread::sleep_for(std::chrono::microseconds(100000));

 //Contract tensors (case 0):
 if(root) std::cout << " Case 0: C=A*B five times: Warm-up: ";
 success = exatn::sync(); assert(success);
 auto time_start = exatn::Timer::timeInSecHR();
 success = exatn::contractTensors("C(i,j)+=A(k,i)*B(k,j)",1.0); assert(success);
 success = exatn::contractTensors("C(i,j)+=A(i,k)*B(k,j)",1.0); assert(success);
 success = exatn::contractTensors("C(i,j)+=A(k,i)*B(j,k)",1.0); assert(success);
 success = exatn::contractTensors("C(i,j)+=A(i,k)*B(j,k)",1.0); assert(success);
 success = exatn::contractTensors("C(i,j)+=A(k,i)*B(k,j)",1.0); assert(success);
 success = exatn::sync(); assert(success);
 auto duration = exatn::Timer::timeInSecHR(time_start);
 if(root) std::cout << "Average performance (GFlop/s) = " << 5.0*2.0*double{DIM}*double{DIM}*double{DIM}/duration/1e9 << std::endl;

 std::this_thread::sleep_for(std::chrono::microseconds(100000));

 //Contract tensors (case 1):
 if(root) std::cout << " Case 1: C=A*B five times: Reuse: ";
 success = exatn::sync(); assert(success);
 time_start = exatn::Timer::timeInSecHR();
 success = exatn::contractTensors("C(i,j)+=A(k,i)*B(k,j)",1.0); assert(success);
 success = exatn::contractTensors("C(i,j)+=A(i,k)*B(k,j)",1.0); assert(success);
 success = exatn::contractTensors("C(i,j)+=A(k,i)*B(j,k)",1.0); assert(success);
 success = exatn::contractTensors("C(i,j)+=A(i,k)*B(j,k)",1.0); assert(success);
 success = exatn::contractTensors("C(i,j)+=A(k,i)*B(k,j)",1.0); assert(success);
 success = exatn::sync(); assert(success);
 duration = exatn::Timer::timeInSecHR(time_start);
 if(root) std::cout << "Average performance (GFlop/s) = " << 5.0*2.0*double{DIM}*double{DIM}*double{DIM}/duration/1e9 << std::endl;

 std::this_thread::sleep_for(std::chrono::microseconds(100000));

 //Contract tensors (case 2):
 if(root) std::cout << " Case 2: C=A*B | F=D*E | I=G*H: Pipeline: ";
 success = exatn::sync(); assert(success);
 time_start = exatn::Timer::timeInSecHR();
 success = exatn::contractTensors("I(i,j)+=G(j,k)*H(i,k)",1.0); assert(success);
 success = exatn::contractTensors("F(i,j)+=D(j,k)*E(i,k)",1.0); assert(success);
 success = exatn::contractTensors("C(i,j)+=A(j,k)*B(i,k)",1.0); assert(success);
 success = exatn::sync(); assert(success);
 duration = exatn::Timer::timeInSecHR(time_start);
 if(root) std::cout << "Average performance (GFlop/s) = " << 3.0*2.0*double{DIM}*double{DIM}*double{DIM}/duration/1e9 << std::endl;

 std::this_thread::sleep_for(std::chrono::microseconds(100000));

 //Contract tensors (case 3):
 if(root) std::cout << " Case 3: I=A*B | I=D*E | I=G*H: Prefetch: ";
 success = exatn::sync(); assert(success);
 time_start = exatn::Timer::timeInSecHR();
 success = exatn::contractTensors("I(i,j)+=G(j,k)*H(i,k)",1.0); assert(success);
 success = exatn::contractTensors("I(i,j)+=D(j,k)*E(i,k)",1.0); assert(success);
 success = exatn::contractTensors("I(i,j)+=A(j,k)*B(i,k)",1.0); assert(success);
 success = exatn::sync(); assert(success);
 duration = exatn::Timer::timeInSecHR(time_start);
 if(root) std::cout << "Average performance (GFlop/s) = " << 3.0*2.0*double{DIM}*double{DIM}*double{DIM}/duration/1e9 << std::endl;

 std::this_thread::sleep_for(std::chrono::microseconds(100000));

 //Destroy tensors:
 if(root) std::cout << " Destroying all tensors ... ";
 success = exatn::destroyTensor("I"); assert(success);
 success = exatn::destroyTensor("H"); assert(success);
 success = exatn::destroyTensor("G"); assert(success);
 success = exatn::destroyTensor("F"); assert(success);
 success = exatn::destroyTensor("E"); assert(success);
 success = exatn::destroyTensor("D"); assert(success);
 success = exatn::destroyTensor("C"); assert(success);
 success = exatn::destroyTensor("B"); assert(success);
 success = exatn::destroyTensor("A"); assert(success);
 if(root) std::cout << "Done\n";

 success = exatn::sync(); assert(success);

 //Create tensors:
 if(root) std::cout << " Creating all tensors ... ";
 success = exatn::createTensor("A",TENS_ELEM_TYPE,TensorShape{DIM,DIM,32ULL}); assert(success);
 success = exatn::createTensor("B",TENS_ELEM_TYPE,TensorShape{DIM,DIM,32ULL}); assert(success);
 success = exatn::createTensor("C",TENS_ELEM_TYPE,TensorShape{DIM,DIM}); assert(success);
 if(root) std::cout << "Done\n";

 //Initialize tensors:
 if(root) std::cout << " Initializing all tensors ... ";
 success = exatn::initTensor("A",1e-4); assert(success);
 success = exatn::initTensor("B",1e-3); assert(success);
 success = exatn::initTensor("C",0.0); assert(success);
 if(root) std::cout << "Done\n";

 //Contract tensors:
 if(root) std::cout << " Case 4: C=A*B: Out-of-core large dims: ";
 success = exatn::sync(); assert(success);
 time_start = exatn::Timer::timeInSecHR();
 success = exatn::contractTensors("C(i,j)+=A(j,k,l)*B(i,k,l)",1.0); assert(success);
 success = exatn::sync(); assert(success);
 duration = exatn::Timer::timeInSecHR(time_start);
 if(root) std::cout << "Average performance (GFlop/s) = " << 2.0*double{DIM}*double{DIM}*double{DIM}*double{32}/duration/1e9 << std::endl;

 std::this_thread::sleep_for(std::chrono::microseconds(100000));

 //Destroy tensors:
 if(root) std::cout << " Destroying all tensors ... ";
 success = exatn::destroyTensor("C"); assert(success);
 success = exatn::destroyTensor("B"); assert(success);
 success = exatn::destroyTensor("A"); assert(success);
 if(root) std::cout << "Done\n";

 success = exatn::sync(); assert(success);

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
 if(root) std::cout << " Case 5: C=A*B out-of-core small dims: ";
 success = exatn::sync(); assert(success);
 time_start = exatn::Timer::timeInSecHR();
 success = exatn::contractTensors("C(c49,c40,c13,c50,c47,c14,c15,c41,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c45,c30,c44,c43,c31,c32,c33,c34,c48,c35,c36,c42,c37,c39,c38,c46)+="
  "A(c49,c40,c13,c50,c62,c47,c14,c15,c63,c41,c64,c65,c16,c66,c67,c68,c69,c17,c18,c19,c70,c71,c72,c73,c74,c75)*"
  "B(c20,c21,c64,c22,c69,c67,c23,c24,c25,c26,c27,c28,c29,c45,c30,c44,c68,c43,c31,c73,c72,c32,c33,c66,c34,c75,c74,c71,c65,c48,c70,c35,c63,c36,c42,c37,c39,c38,c46,c62)",1.0); assert(success);
 success = exatn::sync(); assert(success);
 duration = exatn::Timer::timeInSecHR(time_start);
 if(root) std::cout << "Average performance (GFlop/s) = " << 8.0*1.099512e3/duration << std::endl;

 //Destroy tensors:
 success = exatn::destroyTensor("C"); assert(success);
 success = exatn::destroyTensor("B"); assert(success);
 success = exatn::destroyTensor("A"); assert(success);

 success = exatn::sync(); assert(success); */

 if(root) std::cout << "Tensor decomposition:" << std::endl;
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
 if(root) std::cout << " Final 1-norm of tensor D (should be close to zero) = " << norm1 << std::endl;

 //Destroy tensors:
 success = exatn::destroyTensor("R"); assert(success);
 success = exatn::destroyTensor("L"); assert(success);
 success = exatn::destroyTensor("D"); assert(success);

 //Synchronize ExaTN server:
 success = exatn::syncClean(); assert(success);
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

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

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
 success = exatn::sync(); assert(success);

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
 success = exatn::sync(); assert(success);

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
 //Retrieve scalar via talsh::Tensor::View:
 auto scalar_view = local_copy->getSliceView<exatn::TensorDataType<TENS_ELEM_TYPE>::value>(); //scalar view
 std::cout << "ENERGY value (via tensor view) = " << scalar_view[std::initializer_list<int>{}] << std::endl;
 local_copy.reset();

 //Access a tensor element directly via talsh::Tensor::View:
 local_copy = exatn::getLocalTensor("Z2"); assert(local_copy);
 auto tensor_view = local_copy->getSliceView<exatn::TensorDataType<TENS_ELEM_TYPE>::value>(); //full tensor view
 std::cout << "Z2[2,3,1,0] = " << tensor_view[{2,3,1,0}] << std::endl;
 local_copy.reset();

 //Synchronize ExaTN server:
 success = exatn::sync(); assert(success);

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
 success = exatn::syncClean(all_processes); assert(success);
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

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

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
 success = exatn::syncClean(all_processes); assert(success);
 exatn::resetLoggingLevel(0,0);
}
#endif

#ifdef EXATN_TEST3
TEST(NumServerTester, superEasyNumServer)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorElementType;

 //exatn::resetLoggingLevel(1,2); //debug

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

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
 synced = exatn::syncClean(); assert(synced);
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST4
TEST(NumServerTester, circuitNumServer)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorNetwork;
 using exatn::TensorElementType;

 //exatn::resetLoggingLevel(1,2); //debug

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

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
 auto success = exatn::syncClean(); assert(success);
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST5
TEST(NumServerTester, circuitConjugateNumServer)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorNetwork;
 using exatn::TensorElementType;

 //exatn::resetLoggingLevel(1,2); //debug

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

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
 auto success = exatn::syncClean(); assert(success);
 //Grab a coffee!
}
#endif

#ifdef EXATN_TEST6
TEST(NumServerTester, largeCircuitNumServer)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorNetwork;
 using exatn::TensorElementType;

 //exatn::resetLoggingLevel(1,2); //debug

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

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
   std::cout << "Final result is " << *body_ptr << "\n";
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
 auto success = exatn::syncClean(); assert(success);
 //Grab a coffee!
}
#endif

#ifdef EXATN_TEST7
TEST(NumServerTester, Sycamore8NumServer)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorNetwork;
 using exatn::TensorElementType;

 //exatn::resetLoggingLevel(1,2); //debug

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

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

#ifdef EXATN_TEST8
TEST(NumServerTester, Sycamore12NumServer)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorNetwork;
 using exatn::TensorElementType;

 //exatn::resetLoggingLevel(1,2); //debug

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

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

#ifdef EXATN_TEST9
TEST(NumServerTester, rcsNumServer)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorNetwork;
 using exatn::TensorElementType;

 //exatn::resetLoggingLevel(1,2); //debug

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

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
 auto success = exatn::syncClean(); assert(success);
 //Grab a coffee!
}
#endif

#ifdef EXATN_TEST10
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

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

 const int nbQubits = 32;
 const std::vector<int> qubitTensorDim(nbQubits, 2);
 auto rootTensor = std::make_shared<Tensor>("Root", qubitTensorDim);

 auto & networkBuildFactory = *(exatn::NetworkBuildFactory::get());
 auto builder = networkBuildFactory.createNetworkBuilderShared("MPS"); assert(builder);
 bool success = builder->setParameter("max_bond_dim", 1); assert(success);

 std::cout << "Building MPS tensor network ... " << std::flush;
 auto tensorNetwork = exatn::makeSharedTensorNetwork("QubitRegister", rootTensor, *builder);
 std::cout << "Done\n" << std::flush;
 tensorNetwork->printIt();

 std::cout << "Creating/Initializing MPS tensors ... " << std::flush;
 const std::vector<std::complex<double>> ZERO_TENSOR_BODY {{1.0, 0.0}, {0.0, 0.0}};
 for(auto iter = tensorNetwork->cbegin(); iter != tensorNetwork->cend(); ++iter){
  if(iter->first != 0){
   auto tensor = iter->second.getTensor();
   const auto & tensorName = tensor->getName();
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
 //Connect the original tensor network with its inverse but leave the measure qubit line open:
 for(std::size_t i = 0; i < nbQubits; ++i){
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
 success = exatn::evaluateSync(ket); assert(success);
 std::cout << "Done\n" << std::flush;

 std::cout << "Destroying MPS tensors ... " << std::flush;
 for(auto iter = tensorNetwork->cbegin(); iter != tensorNetwork->cend(); ++iter){
  if(iter->first != 0){
   success = exatn::destroyTensor(iter->second.getTensor()->getName());
   assert(success);
  }
 }
 std::cout << "Done\n" << std::flush;

 success = exatn::syncClean(); assert(success);
}
#endif

#ifdef EXATN_TEST11
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

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 if(used_mem != 0) exatn::numericalServer->printAllocatedTensors();
 assert(used_mem == 0);

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
   auto talsh_tensor = exatn::getLocalTensor(component->network->getTensor(0)->getName());
   const std::complex<double> * body_ptr;
   auto access_granted = talsh_tensor->getDataAccessHostConst(&body_ptr); assert(access_granted);
   std::cout << "Component " << component->network->getTensor(0)->getName() << " expectation value = "
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
  auto success = exatn::syncClean(); assert(success);
 }
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST12
TEST(NumServerTester, IsingTNO)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::TensorNetwork;
 using exatn::TensorOperator;
 using exatn::TensorExpansion;
 using exatn::TensorElementType;

 const auto TENS_ELEM_TYPE = TensorElementType::COMPLEX64;
 const int num_sites = 4;
 const int bond_dim_lim = 4;
 const int max_bond_dim = std::min(static_cast<int>(std::pow(2,num_sites/2)),bond_dim_lim);
 const int arity = 2;
 const std::string tn_type = "TTN"; //MPS or TTN

 //exatn::resetLoggingLevel(1,2); //debug

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

 bool success = true;

 //Define Ising Hamiltonian constants:
 constexpr std::complex<double> ZERO{ 0.0, 0.0};
 constexpr std::complex<double> HAMT{-1.0, 0.0};
 constexpr std::complex<double> HAMU{-2.0, 0.0};

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
 auto ham = exatn::makeSharedTensorOperator("Hamiltonian");
 success = ham->appendComponent(t01,{{0,0},{1,1}},{{0,2},{1,3}},{1.0,0.0}); assert(success);
 success = ham->appendComponent(t12,{{1,0},{2,1}},{{1,2},{2,3}},{1.0,0.0}); assert(success);
 success = ham->appendComponent(t23,{{2,0},{3,1}},{{2,2},{3,3}},{1.0,0.0}); assert(success);
 success = ham->appendComponent(u00,{{0,0}},{{0,1}},{1.0,0.0}); assert(success);
 success = ham->appendComponent(u11,{{1,0}},{{1,1}},{1.0,0.0}); assert(success);
 success = ham->appendComponent(u22,{{2,0}},{{2,1}},{1.0,0.0}); assert(success);
 success = ham->appendComponent(u33,{{3,0}},{{3,1}},{1.0,0.0}); assert(success);
 //ham->printIt(); //debug

 //Configure the tensor network builder:
 auto tn_builder = exatn::getTensorNetworkBuilder(tn_type); assert(tn_builder);
 if(tn_type == "MPS"){
  success = tn_builder->setParameter("max_bond_dim",max_bond_dim); assert(success);
 }else if(tn_type == "TTN"){
  success = tn_builder->setParameter("max_bond_dim",max_bond_dim); assert(success);
  success = tn_builder->setParameter("arity",arity); assert(success);
 }else{
  assert(false);
 }

 //Build a tensor network vector:
 auto ket_tensor = exatn::makeSharedTensor("TensorSpace",std::vector<int>(num_sites,2));
 auto vec_net = exatn::makeSharedTensorNetwork("VectorNet",ket_tensor,*tn_builder,false);
 vec_net->markOptimizableAllTensors();
 //vec_net->printIt(); //debug
 auto vec_tns = exatn::makeSharedTensorExpansion("VectorTNS",vec_net,std::complex<double>{1.0,0.0});
 auto rhs_net = exatn::makeSharedTensorNetwork("RightHandSideNet",ket_tensor,*tn_builder,false);
 auto rhs_tns = exatn::makeSharedTensorExpansion("RightHandSideTNS",rhs_net,std::complex<double>{1.0,0.0});

 //Build a tensor network operator:
 auto space_tensor = exatn::makeSharedTensor("TensorSpaceMap",std::vector<int>(num_sites*2,2));
 auto ham_net = exatn::makeSharedTensorNetwork("HamiltonianNet",space_tensor,*tn_builder,true);
 ham_net->markOptimizableAllTensors();
 //ham_net->printIt(); //debug
 auto ham_tno = exatn::makeSharedTensorOperator("HamiltonianTNO");
 success = ham_tno->appendComponent(ham_net,{{0,0},{1,1},{2,2},{3,3}},{{0,4},{1,5},{2,6},{3,7}},{1.0,0.0}); assert(success);

 {//Numerical evaluation:
  //Create and initialize Hamiltonian tensors:
  std::cout << "Creating Hamiltonian tensors ... ";
  success = exatn::createTensorSync(t01,TENS_ELEM_TYPE); assert(success);
  success = exatn::createTensorSync(t12,TENS_ELEM_TYPE); assert(success);
  success = exatn::createTensorSync(t23,TENS_ELEM_TYPE); assert(success);
  success = exatn::createTensorSync(u00,TENS_ELEM_TYPE); assert(success);
  success = exatn::createTensorSync(u11,TENS_ELEM_TYPE); assert(success);
  success = exatn::createTensorSync(u22,TENS_ELEM_TYPE); assert(success);
  success = exatn::createTensorSync(u33,TENS_ELEM_TYPE); assert(success);
  std::cout << "Ok" << std::endl;

  std::cout << "Initializing Hamiltonian tensors ... ";
  success = exatn::initTensorDataSync("T01",hamt); assert(success);
  success = exatn::initTensorDataSync("T12",hamt); assert(success);
  success = exatn::initTensorDataSync("T23",hamt); assert(success);
  success = exatn::initTensorDataSync("U00",hamu); assert(success);
  success = exatn::initTensorDataSync("U11",hamu); assert(success);
  success = exatn::initTensorDataSync("U22",hamu); assert(success);
  success = exatn::initTensorDataSync("U33",hamu); assert(success);
  std::cout << "Ok" << std::endl;

  //Create and initialize tensor network vector tensors:
  std::cout << "Creating and initializing tensor network vector tensors ... ";
  success = exatn::createTensorsSync(*vec_net,TENS_ELEM_TYPE); assert(success);
  success = exatn::initTensorsRndSync(*vec_net); assert(success);
  success = exatn::createTensorsSync(*rhs_net,TENS_ELEM_TYPE); assert(success);
  success = exatn::initTensorsRndSync(*rhs_net); assert(success);
  std::cout << "Ok" << std::endl;

  //Create and initialize tensor network operator tensors:
  std::cout << "Creating and initializing tensor network operator tensors ... ";
  success = exatn::createTensorsSync(*ham_net,TENS_ELEM_TYPE); assert(success);
  success = exatn::initTensorsRndSync(*ham_net); assert(success);
  std::cout << "Ok" << std::endl;

  //Remap tensor operators as tensor expansions:
  auto ham_expansion = exatn::makeSharedTensorExpansion(*ham,*ket_tensor);
  ham_expansion->printIt(); //debug
  auto ham_tno_expansion = exatn::makeSharedTensorExpansion(*ham_tno,*ket_tensor);
  ham_tno_expansion->printIt(); //debug

  //Create and initialize special tensors in the Hamiltonian tensor expansion:
  std::cout << "Creating and initializing special Hamiltonian tensors ... ";
  for(auto net = ham_expansion->begin(); net != ham_expansion->end(); ++net){
   success = exatn::createTensorsSync(*(net->network),TENS_ELEM_TYPE); assert(success);
   success = exatn::initTensorsSpecialSync(*(net->network)); assert(success);
  }
  std::cout << "Ok" << std::endl;

  //Ground state search for the original Hamiltonian:
  std::cout << "Ground state search for the original Hamiltonian:" << std::endl;
  //success = exatn::sync(); assert(success);
  //success = exatn::balanceNorm2Sync(*vec_tns,1.0,true); assert(success);
  exatn::TensorNetworkOptimizer::resetDebugLevel(1,0);
  exatn::TensorNetworkOptimizer optimizer1(ham,vec_tns,1e-5);
  success = exatn::sync(); assert(success);
  bool converged = optimizer1.optimize();
  success = exatn::sync(); assert(success);
  if(converged){
   std::cout << "Search succeeded!" << std::endl;
  }else{
   std::cout << "Search failed!" << std::endl;
   assert(false);
  }
  const auto expect_val1 = optimizer1.getExpectationValue();

  //Reconstruct the Ising Hamiltonian as a tensor network operator:
  std::cout << "Reconstructing Ising Hamiltonian with a tensor network operator:" << std::endl;
  double ham_norm = 0.0;
  success = exatn::normalizeNorm2Sync(*ham_expansion,1.0,&ham_norm); assert(success);
  std::cout << "Original Hamiltonian operator norm = " << ham_norm << std::endl;
  //success = exatn::balanceNorm2Sync(*ham_tno_expansion,1.0,true); assert(success);
  ham_tno_expansion->conjugate();
  exatn::TensorNetworkReconstructor::resetDebugLevel(1,0); //debug
  exatn::TensorNetworkReconstructor reconstructor(ham_expansion,ham_tno_expansion,1e-6);
  success = exatn::sync(); assert(success);
  double residual_norm, fidelity;
  bool reconstructed = reconstructor.reconstruct(&residual_norm,&fidelity);
  success = exatn::sync(); assert(success);
  if(reconstructed){
   std::cout << "Reconstruction succeeded: Residual norm = " << residual_norm
             << "; Fidelity = " << fidelity << std::endl;
  }else{
   std::cout << "Reconstruction failed!" << std::endl;
   assert(false);
  }
  ham_tno_expansion->conjugate();
  ham_tno_expansion->rescale(std::complex<double>{ham_norm,0.0});
  const auto num_components = ham_tno->getNumComponents();
  assert(ham_tno_expansion->getNumComponents() == num_components);
  for(std::size_t i = 0; i < num_components; ++i){
   (*ham_tno)[i].coefficient = (*ham_tno_expansion)[i].coefficient;
  }

  //Ground state search for the tensor network Hamiltonian:
  std::cout << "Ground state search for the tensor network Hamiltonian:" << std::endl;
  exatn::TensorNetworkOptimizer::resetDebugLevel(1,0);
  exatn::TensorNetworkOptimizer optimizer2(ham_tno,vec_tns,1e-5);
  success = exatn::sync(); assert(success);
  converged = optimizer2.optimize();
  success = exatn::sync(); assert(success);
  if(converged){
   std::cout << "Search succeeded!" << std::endl;
  }else{
   std::cout << "Search failed!" << std::endl;
   assert(false);
  }
  const auto expect_val2 = optimizer2.getExpectationValue();
  std::cout << "Relative eigenvalue error due to reconstruction is "
            << std::abs(expect_val1 - expect_val2) / std::abs(expect_val1) * 1e2 << " %\n";

  //Linear system solver for the tensor network Hamiltonian:
  std::cout << "Linear solver for the tensor network Hamiltonian:" << std::endl;
  success = exatn::initTensorsRndSync(*vec_net); assert(success);
  exatn::TensorNetworkLinearSolver::resetDebugLevel(1,0);
  exatn::TensorNetworkLinearSolver linsolver(ham_tno,rhs_tns,vec_tns,1e-5);
  success = exatn::sync(); assert(success);
  converged = linsolver.solve(&residual_norm,&fidelity);
  success = exatn::sync(); assert(success);
  if(converged){
   std::cout << "Linear solver succeeded: Residual norm = " << residual_norm
             << "; Fidelity = " << fidelity << std::endl;
  }else{
   std::cout << "Linear solver failed!" << std::endl;
   assert(false);
  }

  //Destroy all tensors:
  success = exatn::sync(); assert(success);
  for(auto net = ham_expansion->begin(); net != ham_expansion->end(); ++net){
   success = exatn::destroyTensorsSync(*(net->network)); assert(success);
  }
  success = exatn::destroyTensorsSync(*ham_net); assert(success);
  success = exatn::destroyTensorsSync(*rhs_net); assert(success);
  success = exatn::destroyTensorsSync(*vec_net); assert(success);
  success = exatn::destroyTensorSync("U33"); assert(success);
  success = exatn::destroyTensorSync("U22"); assert(success);
  success = exatn::destroyTensorSync("U11"); assert(success);
  success = exatn::destroyTensorSync("U00"); assert(success);
  success = exatn::destroyTensorSync("T23"); assert(success);
  success = exatn::destroyTensorSync("T12"); assert(success);
  success = exatn::destroyTensorSync("T01"); assert(success);

  //Synchronize:
  success = exatn::syncClean(); assert(success);
 }
 exatn::resetLoggingLevel(0,0);
}
#endif

#ifdef EXATN_TEST13
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

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 if(used_mem != 0) exatn::numericalServer->printAllocatedTensors();
 assert(used_mem == 0);

 auto & networkBuildFactory = *(exatn::NetworkBuildFactory::get());
 auto builder = networkBuildFactory.createNetworkBuilderShared("MPS"); assert(builder);
 bool success = builder->setParameter("max_bond_dim",1); assert(success);
 auto rootTensor = exatn::makeSharedTensor("Root",TensorShape{2,2,2,2});
 auto tensorNetwork = exatn::makeSharedTensorNetwork("QubitRegister", rootTensor, *builder);
 tensorNetwork->printIt();

 success = exatn::createTensorSync(tensorNetwork->getTensor(0), TensorElementType::COMPLEX64);
 assert(success);
 success = exatn::initTensorSync(tensorNetwork->getTensor(0)->getName(), 0.0);
 assert(success);
 const std::vector<std::complex<double>> ZERO_TENSOR_BODY {{1.0, 0.0}, {0.0, 0.0}};
 for(auto iter = tensorNetwork->cbegin(); iter != tensorNetwork->cend(); ++iter){
  if(iter->first != 0){
   auto tensor = iter->second.getTensor();
   const auto & tensorName = tensor->getName();
   success = exatn::createTensorSync(tensor, exatn::TensorElementType::COMPLEX64);
   assert(success);
   success = exatn::initTensorDataSync(tensorName, ZERO_TENSOR_BODY);
   assert(success);
  }
 }
 success = exatn::evaluateSync(*tensorNetwork); assert(success);
 success = exatn::destroyTensorsSync(*tensorNetwork); assert(success);
 success = exatn::syncClean(); assert(success);
}
#endif

#ifdef EXATN_TEST14
TEST(NumServerTester, TestSVD)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::TensorNetwork;
 using exatn::TensorOperator;
 using exatn::TensorExpansion;
 using exatn::TensorElementType;

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

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

 success = exatn::syncClean(); assert(success);
}
#endif

#ifdef EXATN_TEST15
TEST(NumServerTester, ParserExaTN)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::TensorNetwork;
 using exatn::TensorOperator;
 using exatn::TensorExpansion;
 using exatn::TensorElementType;

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

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
 exatn::syncClean();
}
#endif

#ifdef EXATN_TEST16
TEST(NumServerTester, testGarbage) {
 using exatn::TensorShape;
 using exatn::Tensor;
 using exatn::TensorNetwork;
 using exatn::TensorElementType;

 //exatn::resetLoggingLevel(1,2); // debug

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

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
 const int NB_QUBITS = 16;

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
 exatn::syncClean();
 // Grab a beer!
}
#endif

#ifdef EXATN_TEST17
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

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

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

  //Synchronize:
  success = exatn::sync(); assert(success);
 }

 //Open new scope:
 {
  //Create tensors:
  success = exatn::createTensor("D",TENS_ELEM_TYPE,TensorShape{2,2,2}); assert(success);
  success = exatn::createTensor("L",TENS_ELEM_TYPE,TensorShape{2,2,2}); assert(success);
  success = exatn::createTensor("R",TENS_ELEM_TYPE,TensorShape{2,2,2,2}); assert(success);

  //Initialize tensors:
  success = exatn::initTensor("D",std::complex<double>{0.0,0.0}); assert(success);
  success = exatn::initTensor("L",ltens_val); assert(success);
  success = exatn::initTensor("R",rtens_val); assert(success);

  //Contract tensors:
  success = exatn::contractTensorsSync("D(b,c,d)+=L(a,b,c)*R(a,b,c,d)",1.0); assert(success);

  //Destroy tensors:
  success = exatn::destroyTensor("R"); assert(success);
  success = exatn::destroyTensor("L"); assert(success);
  success = exatn::destroyTensor("D"); assert(success);

  //Synchronize:
  success = exatn::syncClean(); assert(success);
 }
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST18
TEST(NumServerTester, neurIPS) {
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::Tensor;
 using exatn::TensorNetwork;
 using exatn::TensorExpansion;
 using exatn::TensorElementType;

 const auto TENS_ELEM_TYPE = TensorElementType::COMPLEX32;

 //exatn::resetLoggingLevel(1,2); //debug
 exatn::activateContrSeqCaching(true);
 //exatn::activateFastMath();

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

 bool success = true;
 bool root = (exatn::getProcessRank() == 0);
 auto builder_mps = exatn::getTensorNetworkBuilder("MPS"); assert(builder_mps);
 auto builder_ttn = exatn::getTensorNetworkBuilder("TTN"); assert(builder_ttn);

 //3:1 1D MERA:
 {
  if(root) std::cout << "Evaluating a 3:1 MERA 1D diagram: ";
  const exatn::DimExtent chi = 18; //Laptop: 18; Summit (4-8 nodes): 64
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

  for(int repeat = 0; repeat < 2; ++repeat){
   success = exatn::sync(); assert(success);
   auto flops = exatn::getTotalFlopCount();
   auto time_start = exatn::Timer::timeInSecHR();
   success = exatn::evaluateTensorNetwork("MERA1d",
    "Z(z0,z1,z2,z3)+=A(a0,a1,a2,z2)*B(b0,b1,b2,z3)*C(c0,c1,a2,b0)*D(d0,d1,a1,c0)*E(e2,e3,d1,c1)*F(a0,d0,e2,z0)*G(e3,b1,b2,z1)");
   assert(success);
   success = exatn::sync(); assert(success);
   auto duration = exatn::Timer::timeInSecHR(time_start);
   flops = exatn::getTotalFlopCount() - flops;
   if(root) std::cout << "Time (s) = " << duration << "; GFlop/s = " << flops/duration/1e9 << std::endl << std::flush;
  }

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
  if(root) std::cout << "Evaluating an AIEM 2:1 TTN diagram: ";
  success = builder_ttn->setParameter("arity",2); assert(success);
  success = builder_ttn->setParameter("max_bond_dim",32); assert(success);
  auto output_tensor_ttn = exatn::makeSharedTensor("Z_TTN",std::vector<exatn::DimExtent>(64,2));
  auto network_ttn = exatn::makeSharedTensorNetwork("TensorTree",output_tensor_ttn,*builder_ttn);
  auto network_ttn_conj = exatn::makeSharedTensorNetwork(*network_ttn,true);
  success = network_ttn_conj->conjugate(); assert(success);
  network_ttn_conj->rename("TensorTreeConj");
  network_ttn->appendTensorGate(exatn::makeSharedTensor("H1",std::vector<exatn::DimExtent>{2,2}),{5});
  network_ttn->appendTensorGate(exatn::makeSharedTensor("H2",std::vector<exatn::DimExtent>{2,2}),{6});
  TensorExpansion ket("ket",network_ttn,{1.0,0.0},true);
  TensorExpansion bra("bra",network_ttn_conj,{1.0,0.0},false);
  TensorExpansion braket(bra,ket);

  std::string deriv_tens_name;
  for(auto tens = network_ttn_conj->cbegin(); tens != network_ttn_conj->cend(); ++tens){
   if(tens->first != 0 && tens->second.getRank() == 3){
    deriv_tens_name = tens->second.getName();
    break;
   }
  }
  TensorExpansion derivative(braket,deriv_tens_name,true);
  //derivative.printIt(); //debug

  for(auto net = derivative.begin(); net != derivative.end(); ++net){
   success = exatn::createTensorsSync(*(net->network),TENS_ELEM_TYPE); assert(success);
  }

  for(auto net = derivative.begin(); net != derivative.end(); ++net){
   success = exatn::initTensorsRndSync(*(net->network)); assert(success);
  }

  auto deriv_output_tensor = derivative[0].network->getTensor(0);
  success = exatn::createTensorSync("acc",TENS_ELEM_TYPE,deriv_output_tensor->getShape()); assert(success);
  success = exatn::initTensorSync("acc",0.0); assert(success);

  for(int repeat = 0; repeat < 2; ++repeat){
   success = exatn::sync(); assert(success);
   auto flops = exatn::getTotalFlopCount();
   auto time_start = exatn::Timer::timeInSecHR();
   success = exatn::evaluate(derivative,exatn::getTensor("acc")); assert(success);
   success = exatn::sync(); assert(success);
   auto duration = exatn::Timer::timeInSecHR(time_start);
   flops = exatn::getTotalFlopCount() - flops;
   if(root) std::cout << "Time (s) = " << duration << "; GFlop/s = " << flops/duration/1e9 << std::endl << std::flush;
  }

  success = exatn::destroyTensorSync("acc"); assert(success);
  for(auto net = derivative.begin(); net != derivative.end(); ++net){
   success = exatn::destroyTensorsSync(*(net->network)); assert(success);
  }
  success = exatn::sync(); assert(success);
 }

 //ML MERA:
 {
  if(root) std::cout << "Evaluating an ML MERA diagram: ";
  const exatn::DimExtent chi1 = 4; //Laptop: 4; Summit (1 node): 10
  const auto chi2 = std::min(chi1*chi1,128ULL);
  const auto chi4 = std::min(chi2*chi2,1024ULL);
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

  for(int repeat = 0; repeat < 2; ++repeat){
   success = exatn::sync(); assert(success);
   auto flops = exatn::getTotalFlopCount();
   auto time_start = exatn::Timer::timeInSecHR();
   success = exatn::evaluateTensorNetwork("ML_MERA",
    "Z(z0,z1,z2)+=A(z0,a1,a2)*B(z1,z2,b2,b3)*C(a1,c1)*D(a2,b2,d2)*E(b3,e1)*F(c1,f1,f2)*G(d2,e1,g2,g3)*H(f1,h1)*I(f2,g2,i2)*J(g3,j1)*K(h1,i2,j1)");
   assert(success);
   success = exatn::sync(); assert(success);
   auto duration = exatn::Timer::timeInSecHR(time_start);
   flops = exatn::getTotalFlopCount() - flops;
   if(root) std::cout << "Time (s) = " << duration << "; GFlop/s = " << flops/duration/1e9 << std::endl << std::flush;
  }

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
 success = exatn::syncClean(); assert(success);
 exatn::deactivateContrSeqCaching();
 exatn::resetLoggingLevel(0,0);
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST19
TEST(NumServerTester, MPSNorm) {
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::Tensor;
 using exatn::TensorNetwork;
 using exatn::TensorElementType;

 const auto TENS_ELEM_TYPE = TensorElementType::COMPLEX32;

 //exatn::resetLoggingLevel(1,2); //debug

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

 bool success = true;

 const int num_qubits = 32;
 auto output_tensor = std::make_shared<Tensor>("Z0", std::vector<unsigned int>(num_qubits, 2));

 std::cout << "Building MPS tensor network ... " << std::flush;
 auto builder = exatn::getTensorNetworkBuilder("MPS"); assert(builder);
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
 double flops = 0.0;
 if(num_qubits == 32){
  std::vector<unsigned int> contr_seq{
   112,47,48,111,49,50,134,46,45,97,134,44,120,41,40,119,43,42,133,38,37,108,133,36,132,33,34,107,132,35,
   131,16,17,94,131,18,106,13,12,105,14,15,122,9,8,121,11,10,130,1,2,115,130,3,69,51,19,129,23,24,
   82,129,25,128,20,21,81,128,22,127,64,62,114,127,63,126,32,30,113,126,31,99,60,61,125,27,28,89,125,29,
   124,55,56,88,124,57,123,52,53,87,123,54,118,121,122,110,119,120,104,7,118,101,115,4,100,113,114,98,111,112,
   96,39,110,95,107,108,93,105,106,92,6,104,91,101,5,90,99,100,86,97,98,85,95,96,84,93,94,83,91,92,
   79,89,90,78,87,88,76,85,86,75,83,84,74,81,82,72,79,59,71,58,78,70,75,76,68,26,74,67,71,72,
   66,69,70,65,67,68,0,65,66
  };
  flops = 7.3569e+10;
  mps_norm.importContractionSequence(contr_seq,flops);
 }else{
  flops = mps_norm.determineContractionSequence("metis");
 }
 std::cout << "Done: Flop count = " << flops << std::endl << std::flush;
 exatn::printContractionSequence(mps_norm.exportContractionSequence());

 std::cout << "Evaluating tensor network ... " << std::flush;
 success = exatn::evaluate(mps_norm); assert(success);
 success = exatn::sync(); assert(success);
 std::cout << "Done" << std::endl << std::flush;

 std::cout << "Destroying tensors ... " << std::flush;
 success = exatn::destroyTensors(mps_norm); assert(success);
 std::cout << "Done" << std::endl << std::flush;

 //Synchronize:
 success = exatn::syncClean(); assert(success);
 exatn::resetLoggingLevel(0,0);
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST20
TEST(NumServerTester, UserDefinedMethod) {
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::Tensor;
 using exatn::TensorNetwork;
 using exatn::TensorElementType;

 const auto TENS_ELEM_TYPE = TensorElementType::REAL64;

 const unsigned int num_occupied = 12;
 const unsigned int num_virtuals = 36;
 const unsigned int num_total = num_occupied + num_virtuals;

 //exatn::resetLoggingLevel(1,2); //debug

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

 bool success = true;

 //Create a tensor:
 success = exatn::createTensor("A",TENS_ELEM_TYPE,TensorShape{num_virtuals,num_virtuals,num_occupied,num_occupied});
 assert(success);

 //Init tensors:
 success = exatn::initTensorRnd("A"); assert(success);

 //Define a user-defined tensor method:
 class MyTensorMethod: public exatn::TensorMethod {
 public:

  MyTensorMethod(unsigned int num_occ,
                 unsigned int num_virt,
                 const std::vector<double> & denominators):
   num_occ_(num_occ), num_virt_(num_virt), denominators_(denominators) {}

  const std::string name() const override {return "MyTensorMethod";}

  const std::string description() const override {return "Division by denominators";}

  void pack(BytePacket & packet) override {} //ignore

  void unpack(BytePacket & packet) override {} //ignore

  int apply(talsh::Tensor & tensor) override {
   //Get access to the tensor body:
   double * body;
   auto access_granted = tensor.getDataAccessHost(&body); assert(access_granted);
   //Retrive tensor dimension extents:
   unsigned int num_dims = 0;
   const auto * tensor_dims = tensor.getDimExtents(num_dims); assert(tensor_dims != nullptr);
   if(num_dims > 0 && num_dims%2 == 0){ //assume first half dims are virt, second half are occ
    //Create a range over all tensor dimensions:
    exatn::TensorRange tens_range(num_dims,tensor_dims);
    //Divide each tensor element by a denominator:
    bool not_over = true;
    while(not_over){
     const auto & multi_index = tens_range.getMultiIndex();
     //Compute the denominator for the current tensor element:
     double den = 0.0;
     for(unsigned int i = 0; i < num_dims/2; ++i) den += denominators_[num_occ_ + multi_index[i]]; //virt
     for(unsigned int i = num_dims/2; i < num_dims; ++i) den -= denominators_[multi_index[i]]; //occ
     body[tens_range.localOffset()] /= den; //divide the tensor element by its denominator
     not_over = tens_range.next(); //proceed to the next tensor element
    }
   }else{
    assert(false);
   }
   return 0;
  }

 private:

  unsigned int num_occ_;
  unsigned int num_virt_;
  std::vector<double> denominators_;
 };

 //Apply the user-defined tensor method to a tensor:
 std::vector<double> denominators(num_total);
 for(unsigned int i = 0; i < num_total; ++i) denominators[i] = static_cast<double>(i+1);
 success = exatn::transformTensor("A",std::shared_ptr<exatn::TensorMethod>(
                                       new MyTensorMethod(num_occupied,num_virtuals,denominators)));
 success = exatn::sync(); assert(success);

 //Destroy tensors:
 success = exatn::destroyTensor("A"); assert(success);

 //Synchronize:
 success = exatn::syncClean(); assert(success);
 exatn::resetLoggingLevel(0,0);
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST21
TEST(NumServerTester, PrintTensors) {
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::Tensor;
 using exatn::TensorNetwork;
 using exatn::TensorElementType;

 const auto TENS_ELEM_TYPE = TensorElementType::REAL64;

 //exatn::resetLoggingLevel(1,2); //debug

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

 bool success = true;

 //Create tensors:
 success = exatn::createTensor("A",TENS_ELEM_TYPE,TensorShape{2,3,4,2}); assert(success);
 success = exatn::createTensor("B",TENS_ELEM_TYPE,TensorShape{2,3,4,2}); assert(success);

 //Init tensors:
 success = exatn::initTensorRnd("A"); assert(success);
 success = exatn::initTensor("B",0.0); assert(success);

 //Print tensor A to screen:
 success = exatn::printTensorSync("A"); assert(success);

 //Print tensor A to file:
 success = exatn::printTensorFileSync("A","tensor.txt"); assert(success);

 //Init tensor B from file:
 success = exatn::initTensorFile("B","tensor.txt"); assert(success);

 //Print tensor B to screen:
 success = exatn::printTensorSync("B"); assert(success);

 //Sync:
 success = exatn::sync(); assert(success);

 //Destroy tensors:
 success = exatn::destroyTensor("B"); assert(success);
 success = exatn::destroyTensor("A"); assert(success);

 //Synchronize:
 success = exatn::syncClean(); assert(success);
 exatn::resetLoggingLevel(0,0);
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST22
TEST(NumServerTester, CollapseTensors) {
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::Tensor;
 using exatn::TensorNetwork;
 using exatn::TensorElementType;

 const auto TENS_ELEM_TYPE = TensorElementType::REAL32;

 //exatn::resetLoggingLevel(1,2); //debug

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

 bool success = true;

 //Test tensor differentiation:
 TensorNetwork network("tensnet");
 auto tensor = exatn::makeSharedTensor("T",TensorShape{16,32});
 success = network.appendTensor(1,tensor,{}); assert(success);
 success = network.appendTensor(2,tensor,{{0,0},{1,1}}); assert(success);
 network.printIt();
 bool deltas_appended = false;
 success = network.differentiateTensor(1,&deltas_appended); assert(success);
 network.printIt();
 success = network.differentiateTensor(2,&deltas_appended); assert(success);
 network.printIt();

 //Test isometric collapse:
 TensorNetwork isonet("isonet");
 auto unitary = exatn::makeSharedTensor("U",TensorShape{8,8});
 auto isometry = exatn::makeSharedTensor("V",TensorShape{16,16,8});
 isometry->registerIsometry({0,1});
 success = isonet.appendTensor(1,unitary,{}); assert(success);
 success = isonet.appendTensor(2,unitary,{{0,0}},{},true); assert(success);
 isonet.printIt();
 TensorNetwork uninet(isonet);
 success = isonet.appendTensor(3,isometry,{{0,2}}); assert(success);
 success = isonet.appendTensor(4,isometry,{{0,2},{1,0},{2,1}},{},true); assert(success);
 isonet.printIt();
 success = isonet.collapseIsometries(); assert(success);
 isonet.printIt();
 unitary->registerIsometry({0});
 unitary->registerIsometry({1});
 success = isonet.collapseIsometries(); assert(success);
 isonet.printIt();
 success = uninet.collapseIsometries(); assert(success);
 uninet.printIt();

 //Synchronize:
 success = exatn::syncClean(); assert(success);
 exatn::resetLoggingLevel(0,0);
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST23
TEST(NumServerTester, Reconstructor) {
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::Tensor;
 using exatn::TensorNetwork;
 using exatn::TensorExpansion;
 using exatn::TensorElementType;

 const auto TENS_ELEM_TYPE = TensorElementType::REAL32;

 //exatn::resetLoggingLevel(1,2); //debug

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);
#ifdef CUQUANTUM
 auto backends = exatn::queryComputationalBackends();
 if(std::find(backends.cbegin(),backends.cend(),"cuquantum") != backends.cend())
  exatn::switchComputationalBackend("cuquantum");
#endif

 bool success = true;

 //Create tensors:
 const int bond_dim = 10;
 success = exatn::createTensor("T",TENS_ELEM_TYPE,TensorShape{200,100}); assert(success);
 success = exatn::createTensor("Z",TENS_ELEM_TYPE,TensorShape{200,100}); assert(success);
 success = exatn::createTensor("A",TENS_ELEM_TYPE,TensorShape{bond_dim,200}); assert(success);
 success = exatn::createTensor("B",TENS_ELEM_TYPE,TensorShape{bond_dim,100}); assert(success);

 //Init tensors:
 success = exatn::initTensorRnd("T"); assert(success);
 success = exatn::initTensor("Z",0.0); assert(success);
 success = exatn::initTensorRnd("A"); assert(success);
 success = exatn::initTensorRnd("B"); assert(success);

 //Compute tensor factors based on the input tensor decomposition (debug):
 //success = exatn::decomposeTensorSVDLR("T(i,j)=A(k,i)*B(k,j)"); assert(success);

 //Construct the approximant tensor network expansion A(k,i)*B(k,j):
 auto approx_net = exatn::makeTensorNetwork("ApproxNet","Z(i,j)=A(k,i)*B(k,j)");
 approx_net->markOptimizableTensors(
             [](const Tensor & tensor){
              return true; //(tensor.getName() == "A");
             });
 auto approximant = exatn::makeSharedTensorExpansion();
 approximant->appendComponent(approx_net,{1.0,0.0});
 approximant->conjugate();
 approximant->rename("Approximant");
 //approximant->printIt(); //debug

 //Construct the target tensor network expansion T(i,j):
 auto target_net = exatn::makeSharedTensorNetwork("TargetNet");
 target_net->appendTensor(1,exatn::getTensor("T"),{});
 auto target = exatn::makeSharedTensorExpansion();
 target->appendComponent(target_net,{1.0,0.0});
 target->rename("Target");
 //target->printIt(); //debug

 //Normalize input tensors in the tensor network expansions to 1.0:
 success = exatn::balanceNorm2Sync(*target,1.0,false); assert(success);
 //success = exatn::balanceNorm2Sync(*approximant,1.0,true); assert(success);

 //Construct the reconstructor (solver):
 exatn::TensorNetworkReconstructor::resetDebugLevel(1); //debug
 exatn::TensorNetworkReconstructor reconstructor(target,approximant,1e-4);

 //Run the reconstructor:
 success = exatn::sync(); assert(success);
 double residual_norm, fidelity;
 bool reconstructed = reconstructor.reconstruct(&residual_norm,&fidelity);
 success = exatn::sync(); assert(success);
 if(reconstructed){
  std::cout << "Reconstruction succeeded: Residual norm = " << residual_norm
            << "; Fidelity = " << fidelity << std::endl;
 }else{
  std::cout << "Reconstruction failed!" << std::endl; assert(false);
 }

 //Destroy tensors:
 success = exatn::destroyTensor("B"); assert(success);
 success = exatn::destroyTensor("A"); assert(success);
 success = exatn::destroyTensor("Z"); assert(success);
 success = exatn::destroyTensor("T"); assert(success);

 //Synchronize:
 success = exatn::syncClean(); assert(success);
 exatn::resetLoggingLevel(0,0);
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST24
TEST(NumServerTester, OptimizerTransverseIsing) {
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::Tensor;
 using exatn::TensorNetwork;
 using exatn::TensorExpansion;
 using exatn::TensorOperator;
 using exatn::TensorElementType;

 const auto TENS_ELEM_TYPE = TensorElementType::REAL32;

 //exatn::resetLoggingLevel(1,2); //debug

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);
#ifdef CUQUANTUM
 auto backends = exatn::queryComputationalBackends();
 if(std::find(backends.cbegin(),backends.cend(),"cuquantum") != backends.cend())
  exatn::switchComputationalBackend("cuquantum");
#endif

 const int num_sites = 4, max_bond_dim = std::pow(2,num_sites/2);
 double g_factor = 1e-1; // >0.0, 1e0 is critical state

 bool success = true;

 //Define, create and initialize Pauli tensors:
 auto pauli_x = exatn::makeSharedTensor("PauliX",TensorShape{2,2});
 auto pauli_z = exatn::makeSharedTensor("PauliZ",TensorShape{2,2});
 success = exatn::createTensor(pauli_x,TENS_ELEM_TYPE); assert(success);
 success = exatn::createTensor(pauli_z,TENS_ELEM_TYPE); assert(success);
 success = exatn::initTensorData("PauliX",std::vector<float>{
                                           0.0f,  1.0f,
                                           1.0f,  0.0f}); assert(success);
 success = exatn::initTensorData("PauliZ",std::vector<float>{
                                           1.0f,  0.0f,
                                           0.0f, -1.0f}); assert(success);

 //Define the tensor network operator:
 auto ising_zz = exatn::makeSharedTensorNetwork("IsingZZ");
 success = ising_zz->appendTensor(1,pauli_z,{}); assert(success);
 success = ising_zz->appendTensor(2,pauli_z,{}); assert(success);
 //ising_zz->printIt(); //debug
 auto ising_x = exatn::makeSharedTensorNetwork("IsingX");
 success = ising_x->appendTensor(1,pauli_x,{}); assert(success);
 //ising_x->printIt(); //debug
 auto ising = exatn::makeSharedTensorOperator("IsingHamiltonian");
 for(int i = 0; i < (num_sites - 1); ++i){
  success = ising->appendComponent(ising_zz, {{i,0},{i+1,2}}, {{i,1},{i+1,3}}, {-1.0,0.0}); assert(success);
 }
 for(int i = 0; i < num_sites; ++i){
  success = ising->appendComponent(ising_x, {{i,0}}, {{i,1}}, {-1.0*g_factor,0.0}); assert(success);
 }
 //ising->printIt(); //debug

 //Create tensor network ansatz:
 auto ansatz_tensor = exatn::makeSharedTensor("AnsatzTensor",std::vector<int>(num_sites,2));
 auto mps_builder = exatn::getTensorNetworkBuilder("MPS"); assert(mps_builder);
 success = mps_builder->setParameter("max_bond_dim",max_bond_dim); assert(success);
 auto ansatz_net = exatn::makeSharedTensorNetwork("Ansatz",ansatz_tensor,*mps_builder);
 ansatz_net->markOptimizableTensors([](const Tensor & tensor){return true;});
 auto ansatz = exatn::makeSharedTensorExpansion();
 ansatz->rename("Ansatz");
 success = ansatz->appendComponent(ansatz_net,{1.0,0.0}); assert(success);
 //ansatz->printIt(); //debug

 //Allocate/initialize tensors in the tensor network ansatz:
 for(auto tens_conn = ansatz_net->begin(); tens_conn != ansatz_net->end(); ++tens_conn){
  if(tens_conn->first != 0){ //input tensors only
   success = exatn::createTensor(tens_conn->second.getTensor(),TENS_ELEM_TYPE); assert(success);
   success = exatn::initTensorRnd(tens_conn->second.getName()); assert(success);
  }
 }
 success = exatn::balanceNorm2Sync(*ansatz,1.0,true); assert(success);

 //Create the full tensor ansatz:
 success = exatn::createTensor(ansatz_tensor,TENS_ELEM_TYPE); assert(success);
 success = exatn::initTensorRnd(ansatz_tensor->getName()); assert(success);
 auto ansatz_full_net = exatn::makeSharedTensorNetwork("AnsatzFull");
 success = ansatz_full_net->appendTensor(1,ansatz_tensor,{}); assert(success);
 ansatz_full_net->markOptimizableAllTensors();
 auto ansatz_full = exatn::makeSharedTensorExpansion();
 ansatz_full->rename("AnsatzFull");
 success = ansatz_full->appendComponent(ansatz_full_net,{1.0,0.0}); assert(success);
 //ansatz_full->printIt(); //debug

 //Perform ground state optimization in a complete tensor space:
 {
  std::cout << "Ground state optimization in the complete tensor space:" << std::endl;
  exatn::TensorNetworkOptimizer::resetDebugLevel(1);
  exatn::TensorNetworkOptimizer optimizer(ising,ansatz_full,1e-4);
  success = exatn::sync(); assert(success);
  bool converged = optimizer.optimize();
  success = exatn::sync(); assert(success);
  if(converged){
   std::cout << "Optimization succeeded!" << std::endl;
   /*std::cout << "Reconstructing ground state on a tensor network manifold:" << std::endl;
   ansatz->conjugate();
   exatn::TensorNetworkReconstructor::resetDebugLevel(1);
   exatn::TensorNetworkReconstructor reconstructor(ansatz_full,ansatz,1e-5);
   double residual_norm, fidelity;
   bool reconstructed = reconstructor.reconstruct(&residual_norm,&fidelity);
   success = exatn::sync(); assert(success);
   ansatz->conjugate();
   if(reconstructed){
    std::cout << "Reconstruction succeeded: Residual norm = " << residual_norm
              << "; Fidelity = " << fidelity << std::endl;
   }else{
    std::cout << "Reconstruction failed!" << std::endl; assert(false);
   }*/
  }else{
   std::cout << "Optimization failed!" << std::endl; assert(false);
  }
 }
 success = exatn::normalizeNorm2Sync(ansatz_tensor->getName(),1.0); assert(success);
 success = exatn::printTensor(ansatz_tensor->getName()); assert(success);
 success = exatn::initTensorSync(ansatz_tensor->getName(),0.0); assert(success);

 //Perform ground state optimization on a tensor network manifold:
 {
  std::cout << "Ground state optimization on a tensor network manifold:" << std::endl;
  exatn::TensorNetworkOptimizer::resetDebugLevel(1);
  exatn::TensorNetworkOptimizer optimizer(ising,ansatz,1e-4);
  optimizer.resetMicroIterations(1);
  bool converged = optimizer.optimize();
  success = exatn::sync(); assert(success);
  if(converged){
   std::cout << "Optimization succeeded!" << std::endl;
   success = exatn::evaluateSync(*((*ansatz)[0].network)); assert(success);
   success = exatn::normalizeNorm2Sync((*ansatz)[0].network->getTensor(0)->getName(),1.0); assert(success);
   success = exatn::printTensor((*ansatz)[0].network->getTensor(0)->getName()); assert(success);
  }else{
   std::cout << "Optimization failed!" << std::endl; assert(false);
  }
 }

 //Destroy tensors:
 success = exatn::destroyTensor(ansatz_tensor->getName()); assert(success);
 success = exatn::destroyTensors(*ansatz_net); assert(success);
 success = exatn::destroyTensor("PauliZ"); assert(success);
 success = exatn::destroyTensor("PauliX"); assert(success);

 //Synchronize:
 success = exatn::syncClean(); assert(success);
 exatn::resetLoggingLevel(0,0);
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST25
TEST(NumServerTester, OptimizerHubbard) {
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::Tensor;
 using exatn::TensorNetwork;
 using exatn::TensorExpansion;
 using exatn::TensorOperator;
 using exatn::TensorElementType;

 const auto TENS_ELEM_TYPE = TensorElementType::COMPLEX32;

 //exatn::resetLoggingLevel(1,2); //debug
 //exatn::resetExecutionSerialization(true,true); //debug

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);
#ifdef CUQUANTUM
 auto backends = exatn::queryComputationalBackends();
 if(std::find(backends.cbegin(),backends.cend(),"cuquantum") != backends.cend())
  exatn::switchComputationalBackend("cuquantum");
#endif

 const int num_sites = 4, max_bond_dim = std::pow(2,num_sites/2);

 bool success = true;

 //Define, create and initialize Pauli tensors:
 auto pauli_x = exatn::makeSharedTensor("PauliX",TensorShape{2,2});
 auto pauli_y = exatn::makeSharedTensor("PauliY",TensorShape{2,2});
 auto pauli_z = exatn::makeSharedTensor("PauliZ",TensorShape{2,2});
 auto pauli_e = exatn::makeSharedTensor("PauliE",TensorShape{2,2});
 success = exatn::createTensor(pauli_x,TENS_ELEM_TYPE); assert(success);
 success = exatn::createTensor(pauli_y,TENS_ELEM_TYPE); assert(success);
 success = exatn::createTensor(pauli_z,TENS_ELEM_TYPE); assert(success);
 success = exatn::createTensor(pauli_e,TENS_ELEM_TYPE); assert(success);
 success = exatn::initTensorData("PauliX",std::vector<std::complex<float>>{
                                           {0.0f,0.0f}, {1.0f,0.0f},
                                           {1.0f,0.0f}, {0.0f,0.0f}}); assert(success);
 success = exatn::initTensorData("PauliY",std::vector<std::complex<float>>{
                                           {0.0f,0.0f}, {1.0f,0.0f},
                                           {1.0f,0.0f}, {0.0f,0.0f}}); assert(success);
 /*success = exatn::initTensorData("PauliY",std::vector<std::complex<float>>{
                                           {0.0f,0.0f}, {0.0f,-1.0f},
                                           {0.0f,1.0f}, {0.0f,0.0f}}); assert(success);*/
 success = exatn::initTensorData("PauliZ",std::vector<std::complex<float>>{
                                           {1.0f,0.0f}, {0.0f,0.0f},
                                           {0.0f,0.0f}, {-1.0f,0.0f}}); assert(success);
 success = exatn::initTensorData("PauliE",std::vector<std::complex<float>>{
                                           {1.0f,0.0f}, {0.0f,0.0f},
                                           {0.0f,0.0f}, {1.0f,0.0f}}); assert(success);

 //Define the tensor network operator:
 auto hubbard = exatn::makeSharedTensorOperator("HubbardHamiltonian");
 auto eeee = exatn::makeSharedTensorNetwork("EEEE");
 success = eeee->appendTensor(1,pauli_e,{}); assert(success);
 success = eeee->appendTensor(2,pauli_e,{}); assert(success);
 success = eeee->appendTensor(3,pauli_e,{}); assert(success);
 success = eeee->appendTensor(4,pauli_e,{}); assert(success);
 success = hubbard->appendComponent(eeee, {{0,0},{1,2}, {2,4}, {3,6}},
                                          {{0,1},{1,3}, {2,5}, {3,7}}, {-1.5,0.0}); assert(success);
 auto xzxe = exatn::makeSharedTensorNetwork("XZXE");
 success = xzxe->appendTensor(1,pauli_x,{}); assert(success);
 success = xzxe->appendTensor(2,pauli_z,{}); assert(success);
 success = xzxe->appendTensor(3,pauli_x,{}); assert(success);
 success = xzxe->appendTensor(4,pauli_e,{}); assert(success);
 success = hubbard->appendComponent(xzxe, {{0,0},{1,2}, {2,4}, {3,6}},
                                          {{0,1},{1,3}, {2,5}, {3,7}}, {-0.5,0.0}); assert(success);
 auto yzye = exatn::makeSharedTensorNetwork("YZYE");
 success = yzye->appendTensor(1,pauli_y,{}); assert(success);
 success = yzye->appendTensor(2,pauli_z,{}); assert(success);
 success = yzye->appendTensor(3,pauli_y,{}); assert(success);
 success = yzye->appendTensor(4,pauli_e,{}); assert(success);
 success = hubbard->appendComponent(yzye, {{0,0},{1,2}, {2,4}, {3,6}},
                                          {{0,1},{1,3}, {2,5}, {3,7}}, {-0.5,0.0}); assert(success);
 auto zzee = exatn::makeSharedTensorNetwork("ZZEE");
 success = zzee->appendTensor(1,pauli_z,{}); assert(success);
 success = zzee->appendTensor(2,pauli_z,{}); assert(success);
 success = zzee->appendTensor(3,pauli_e,{}); assert(success);
 success = zzee->appendTensor(4,pauli_e,{}); assert(success);
 success = hubbard->appendComponent(zzee, {{0,0},{1,2}, {2,4}, {3,6}},
                                          {{0,1},{1,3}, {2,5}, {3,7}}, {0.75,0.0}); assert(success);
 auto exzx = exatn::makeSharedTensorNetwork("EXZX");
 success = exzx->appendTensor(1,pauli_e,{}); assert(success);
 success = exzx->appendTensor(2,pauli_x,{}); assert(success);
 success = exzx->appendTensor(3,pauli_z,{}); assert(success);
 success = exzx->appendTensor(4,pauli_x,{}); assert(success);
 success = hubbard->appendComponent(exzx, {{0,0},{1,2}, {2,4}, {3,6}},
                                          {{0,1},{1,3}, {2,5}, {3,7}}, {-0.5,0.0}); assert(success);
 auto eyzy = exatn::makeSharedTensorNetwork("EYZY");
 success = eyzy->appendTensor(1,pauli_e,{}); assert(success);
 success = eyzy->appendTensor(2,pauli_y,{}); assert(success);
 success = eyzy->appendTensor(3,pauli_z,{}); assert(success);
 success = eyzy->appendTensor(4,pauli_y,{}); assert(success);
 success = hubbard->appendComponent(eyzy, {{0,0},{1,2}, {2,4}, {3,6}},
                                          {{0,1},{1,3}, {2,5}, {3,7}}, {-0.5,0.0}); assert(success);
 auto eezz = exatn::makeSharedTensorNetwork("EEZZ");
 success = eezz->appendTensor(1,pauli_e,{}); assert(success);
 success = eezz->appendTensor(2,pauli_e,{}); assert(success);
 success = eezz->appendTensor(3,pauli_z,{}); assert(success);
 success = eezz->appendTensor(4,pauli_z,{}); assert(success);
 success = hubbard->appendComponent(eezz, {{0,0},{1,2}, {2,4}, {3,6}},
                                          {{0,1},{1,3}, {2,5}, {3,7}}, {0.75,0.0}); assert(success);
 //hubbard->printIt(); //debug

 //Create tensor network ansatz:
 auto ansatz_tensor = exatn::makeSharedTensor("AnsatzTensor",std::vector<int>(num_sites,2));
 auto mps_builder = exatn::getTensorNetworkBuilder("MPS"); assert(mps_builder);
 success = mps_builder->setParameter("max_bond_dim",max_bond_dim); assert(success);
 auto ansatz_net = exatn::makeSharedTensorNetwork("Ansatz",ansatz_tensor,*mps_builder);
 ansatz_net->markOptimizableTensors([](const Tensor & tensor){return true;});
 auto ansatz = exatn::makeSharedTensorExpansion();
 ansatz->rename("Ansatz");
 success = ansatz->appendComponent(ansatz_net,{1.0,0.0}); assert(success);
 //ansatz->printIt(); //debug

 //Allocate/initialize tensors in the tensor network ansatz:
 for(auto tens_conn = ansatz_net->begin(); tens_conn != ansatz_net->end(); ++tens_conn){
  if(tens_conn->first != 0){ //input tensors only
   success = exatn::createTensor(tens_conn->second.getTensor(),TENS_ELEM_TYPE); assert(success);
   success = exatn::initTensorRnd(tens_conn->second.getName()); assert(success);
   //success = exatn::initTensor(tens_conn->second.getName(),1e-3f); assert(success);
  }
 }
 success = exatn::balanceNorm2Sync(*ansatz,1.0,true); assert(success);

 //Create the full tensor ansatz:
 success = exatn::createTensor(ansatz_tensor,TENS_ELEM_TYPE); assert(success);
 success = exatn::initTensorRnd(ansatz_tensor->getName()); assert(success);
 auto ansatz_full_net = exatn::makeSharedTensorNetwork("AnsatzFull");
 success = ansatz_full_net->appendTensor(1,ansatz_tensor,{}); assert(success);
 ansatz_full_net->markOptimizableAllTensors();
 auto ansatz_full = exatn::makeSharedTensorExpansion();
 ansatz_full->rename("AnsatzFull");
 success = ansatz_full->appendComponent(ansatz_full_net,{1.0,0.0}); assert(success);
 //ansatz_full->printIt(); //debug

 //Perform ground state optimization in a complete tensor space:
 {
  std::cout << "Ground state optimization in the complete tensor space:" << std::endl;
  exatn::TensorNetworkOptimizer::resetDebugLevel(1);
  exatn::TensorNetworkOptimizer optimizer(hubbard,ansatz_full,1e-4);
  success = exatn::sync(); assert(success);
  bool converged = optimizer.optimize();
  success = exatn::sync(); assert(success);
  if(converged){
   std::cout << "Optimization succeeded!" << std::endl;
  }else{
   std::cout << "Optimization failed!" << std::endl; assert(false);
  }
 }
 success = exatn::normalizeNorm2Sync(ansatz_tensor->getName(),1.0); assert(success);
 success = exatn::printTensor(ansatz_tensor->getName()); assert(success);
 success = exatn::initTensorSync(ansatz_tensor->getName(),0.0); assert(success);
#if 0
 //Perform ground state optimization on a tensor network manifold:
 {
  std::cout << "Ground state optimization on a tensor network manifold:" << std::endl;
  exatn::TensorNetworkOptimizer::resetDebugLevel(1);
  exatn::TensorNetworkOptimizer optimizer(hubbard,ansatz,1e-4);
  optimizer.resetMicroIterations(1);
  bool converged = optimizer.optimize();
  success = exatn::sync(); assert(success);
  if(converged){
   std::cout << "Optimization succeeded!" << std::endl;
   success = exatn::evaluateSync(*((*ansatz)[0].network)); assert(success);
   success = exatn::normalizeNorm2Sync((*ansatz)[0].network->getTensor(0)->getName(),1.0); assert(success);
   success = exatn::printTensor((*ansatz)[0].network->getTensor(0)->getName()); assert(success);
  }else{
   std::cout << "Optimization failed!" << std::endl; assert(false);
  }
 }
#endif
 //Destroy tensors:
 success = exatn::destroyTensor(ansatz_tensor->getName()); assert(success);
 success = exatn::destroyTensors(*ansatz_net); assert(success);
 success = exatn::destroyTensor("PauliE"); assert(success);
 success = exatn::destroyTensor("PauliZ"); assert(success);
 success = exatn::destroyTensor("PauliY"); assert(success);
 success = exatn::destroyTensor("PauliX"); assert(success);

 //Synchronize:
 success = exatn::syncClean(); assert(success);
 exatn::resetLoggingLevel(0,0);
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST26
TEST(NumServerTester, ExaTNGenVisitor) {
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::Tensor;
 using exatn::TensorComposite;
 using exatn::TensorNetwork;
 using exatn::TensorExpansion;
 using exatn::TensorOperator;
 using exatn::TensorElementType;

 const auto TENS_ELEM_TYPE = TensorElementType::COMPLEX64;

 //Test configuration:
 const int bond_dim_lim = 4;
 const int num_sites = 16;
 const int max_bond_dim = std::min(static_cast<int>(std::pow(2,num_sites/2)),bond_dim_lim);
 const int max_layers = (num_sites - 1); //1 less CNOT gates
 bool EVALUATE_FULL_TENSOR = false;

 exatn::resetLoggingLevel(1,1); //debug
 //exatn::resetContrSeqOptimizer("cutnn");

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

 bool success = true;

 //Define the initial qubit state vector:
 std::vector<std::complex<double>> qzero {
  {1.0,0.0}, {0.0,0.0}
 };

 //Create a shallow quantum circuit:
 // Create and initialize qubit tensors:
 for(unsigned int i = 0; i < num_sites; ++i){
  success = exatn::createTensor("Q"+std::to_string(i),TENS_ELEM_TYPE,TensorShape{2}); assert(success);
 }
 for(unsigned int i = 0; i < num_sites; ++i){
  success = exatn::initTensorData("Q"+std::to_string(i),qzero); assert(success);
 }
 // Create and initalize 1-body gates:
 success = exatn::createTensor("H",TENS_ELEM_TYPE,TensorShape{2,2}); assert(success);
 success = exatn::initTensorData("H",exatn::quantum::getGateData(exatn::quantum::Gate::gate_H)); assert(success);
 // Create and initialize 2-body gates:
 success = exatn::createTensor("CNOT",TENS_ELEM_TYPE,TensorShape{2,2,2,2}); assert(success);
 success = exatn::initTensorData("CNOT",exatn::quantum::getGateData(exatn::quantum::Gate::gate_CX)); assert(success);
 // Append qubit tensors, 1-body and 2-body gates to a tensor network:
 auto circuit_net = exatn::makeSharedTensorNetwork("Circuit");
 for(unsigned int i = 0; i < num_sites; ++i){
  success = circuit_net->appendTensor(i+1,exatn::getTensor("Q"+std::to_string(i)),{}); assert(success);
 }
 for(unsigned int i = 0; i < 1; ++i){
  success = circuit_net->appendTensorGate(exatn::getTensor("H"),{i});
 }
 for(unsigned int i = 1; i < std::min(max_layers+1,num_sites); ++i){
  success = circuit_net->appendTensorGate(exatn::getTensor("CNOT"),{i,i-1}); assert(success);
 }
 auto circuit = exatn::makeSharedTensorExpansion("Circuit",circuit_net,std::complex<double>{1.0,0.0});
 //success = exatn::balanceNormalizeNorm2Sync(*circuit,1.0,1.0,false); assert(success);
 //circuit->printIt(); //debug

 //Evaluate the quantum circuit:
 if(EVALUATE_FULL_TENSOR){
  success = exatn::evaluateSync(*circuit_net); assert(success);
 }

 //Create tensor network ansatz:
 auto mps_builder = exatn::getTensorNetworkBuilder("MPS"); assert(mps_builder);
 success = mps_builder->setParameter("max_bond_dim",max_bond_dim); assert(success);
 auto ansatz_tensor = exatn::makeSharedTensor("AnsatzTensor",std::vector<int>(num_sites,2));
 auto ansatz_net = exatn::makeSharedTensorNetwork("Ansatz",ansatz_tensor,*mps_builder);
 ansatz_net->markOptimizableTensors([](const Tensor & tensor){return true;});
 auto ansatz = exatn::makeSharedTensorExpansion("Ansatz",ansatz_net,std::complex<double>{1.0,0.0});
 for(auto tens_conn = ansatz_net->begin(); tens_conn != ansatz_net->end(); ++tens_conn){
  if(tens_conn->first != 0){ //input tensors only
   success = exatn::createTensor(tens_conn->second.getTensor(),TENS_ELEM_TYPE); assert(success);
   success = exatn::initTensorRnd(tens_conn->second.getName()); assert(success);
  }
 }
 success = exatn::balanceNormalizeNorm2Sync(*ansatz,1.0,1.0,true); assert(success);
 //ansatz->printIt(); //debug

 //Create the full tensor ansatz:
 auto ansatz_full_tensor = exatn::makeSharedTensor("AnsatzFullTensor",std::vector<int>(num_sites,2));
 auto ansatz_full_net = exatn::makeSharedTensorNetwork("AnsatzFull");
 success = ansatz_full_net->appendTensor(1,ansatz_full_tensor,{}); assert(success);
 ansatz_full_net->markOptimizableAllTensors();
 auto ansatz_full = exatn::makeSharedTensorExpansion("AnsatzFull",ansatz_full_net,std::complex<double>{1.0,0.0});
 //ansatz_full->printIt(); //debug
 if(EVALUATE_FULL_TENSOR){
  success = exatn::createTensor(ansatz_full_tensor,TENS_ELEM_TYPE); assert(success);
  success = exatn::initTensorRnd(ansatz_full_tensor->getName()); assert(success);
  success = exatn::balanceNormalizeNorm2Sync(*ansatz_full,1.0,1.0,true); assert(success);
 }

 //Reconstruct the quantum circuit by a given tensor network ansatz:
 std::cout << "Reconstructing the quantum circuit by a given tensor network ansatz:" << std::endl;
 ansatz->conjugate();
 exatn::TensorNetworkReconstructor::resetDebugLevel(1); //debug
 exatn::TensorNetworkReconstructor reconstructor(circuit,ansatz,1e-5);
 reconstructor.resetLearningRate(1.0);
 success = exatn::sync(); assert(success);
 double residual_norm, fidelity;
 bool reconstructed = reconstructor.reconstruct(&residual_norm,&fidelity);
 success = exatn::sync(); assert(success);
 if(reconstructed){
  std::cout << "Reconstruction succeeded: Residual norm = " << residual_norm
            << "; Fidelity = " << fidelity << std::endl;
  /*for(auto tens = ansatz_net->cbegin(); tens != ansatz_net->cend(); ++tens){
   if(tens->first != 0) exatn::printTensor(tens->second.getName()); //debug
  }*/
  if(EVALUATE_FULL_TENSOR){
   const auto & tensor0_name = ansatz_full_tensor->getName();
   const auto & tensor1_name = circuit_net->getTensor(0)->getName();
   success = exatn::initTensorSync(tensor0_name,0.0); assert(success);
   success = exatn::evaluateSync(*ansatz,ansatz_full_tensor); assert(success);
   success = exatn::normalizeNorm2Sync(tensor0_name); assert(success);
   success = exatn::normalizeNorm2Sync(tensor1_name); assert(success);
   std::string addition;
   success = exatn::generate_addition_pattern(ansatz_full_tensor->getRank(),addition,false,tensor0_name,tensor1_name); assert(success);
   success = exatn::addTensors(addition,-1.0); assert(success);
   double norm2 = 0.0;
   success = exatn::computeNorm2Sync(tensor0_name,norm2);
   std::cout << "2-norm of the tensor-difference = " << norm2 << std::endl;
   //std::cout << "Full circuit tensor (reference):" << std::endl; //debug
   //exatn::printTensor(tensor1_name); //debug
  }
 }else{
  std::cout << "Reconstruction failed!" << std::endl;
  assert(false);
 }
 ansatz->conjugate();

 //Destroy tensors:
 if(EVALUATE_FULL_TENSOR){
  success = exatn::destroyTensor(ansatz_full_tensor->getName()); assert(success);
  success = exatn::destroyTensor(ansatz_tensor->getName()); assert(success);
 }
 success = exatn::destroyTensors(*ansatz_net); assert(success);
 success = exatn::destroyTensors(*circuit_net); assert(success);

 //Synchronize:
 success = exatn::syncClean(); assert(success);
 exatn::resetLoggingLevel(0,0);
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST27
TEST(NumServerTester, HubbardHamiltonian) {
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::Tensor;
 using exatn::TensorComposite;
 using exatn::TensorNetwork;
 using exatn::TensorExpansion;
 using exatn::TensorOperator;
 using exatn::TensorElementType;

 const auto TENS_ELEM_TYPE = TensorElementType::COMPLEX64;

 //exatn::resetLoggingLevel(1,2); //debug

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

 bool success = true;

 const int num_sites = 8, max_bond_dim = std::min(static_cast<int>(std::pow(2,num_sites/2)),16); //2x2 sites x dim(4) = 8 qubits (sites)

 //Read 2x2 Hubbard Hamiltonian in spin representation:
 //auto hubbard_operator = exatn::quantum::readSpinHamiltonian("MCVQEHam","mcvqe_8q.ofn.txt",TENS_ELEM_TYPE,"OpenFermion");
 //auto hubbard_operator = exatn::quantum::readSpinHamiltonian("MCVQEHam","mcvqe_8q.qcw.txt",TENS_ELEM_TYPE,"QCWare");
 //success = hubbard_operator->deleteComponent(0); assert(success);
 auto hubbard_operator = exatn::quantum::readSpinHamiltonian("HubbardHam","hubbard_2x2_8q.ofn.txt",TENS_ELEM_TYPE,"OpenFermion");
 hubbard_operator->printIt();

 //Create tensor network ansatz:
 auto mps_builder = exatn::getTensorNetworkBuilder("MPS"); assert(mps_builder);
 success = mps_builder->setParameter("max_bond_dim",max_bond_dim); assert(success);
 auto ansatz_tensor = exatn::makeSharedTensor("AnsatzTensor",std::vector<int>(num_sites,2));
 auto ansatz_net = exatn::makeSharedTensorNetwork("Ansatz",ansatz_tensor,*mps_builder);
 ansatz_net->markOptimizableTensors([](const Tensor & tensor){return true;});
 auto ansatz = exatn::makeSharedTensorExpansion("Ansatz",ansatz_net,std::complex<double>{1.0,0.0});
 //ansatz->printIt(); //debug

 //Allocate/initialize tensors in the tensor network ansatz:
 for(auto tens_conn = ansatz_net->begin(); tens_conn != ansatz_net->end(); ++tens_conn){
  if(tens_conn->first != 0){ //input tensors only
   success = exatn::createTensorSync(tens_conn->second.getTensor(),TENS_ELEM_TYPE); assert(success);
   success = exatn::initTensorRndSync(tens_conn->second.getName()); assert(success);
  }
 }
 success = exatn::balanceNormalizeNorm2Sync(*ansatz,1.0,1.0,true); assert(success);

 //Create the full tensor ansatz:
 success = exatn::createTensorSync(ansatz_tensor,TENS_ELEM_TYPE); assert(success);
 success = exatn::initTensorRndSync(ansatz_tensor->getName()); assert(success);
 auto ansatz_full_net = exatn::makeSharedTensorNetwork("AnsatzFull");
 success = ansatz_full_net->appendTensor(1,ansatz_tensor,{}); assert(success);
 ansatz_full_net->markOptimizableAllTensors();
 auto ansatz_full = exatn::makeSharedTensorExpansion("AnsatzFull",ansatz_full_net,std::complex<double>{1.0,0.0});
 //ansatz_full->printIt(); //debug

 //Perform ground state optimization in a complete tensor space:
 {
  std::cout << "Ground state optimization in the complete tensor space:" << std::endl;
  exatn::TensorNetworkOptimizer::resetDebugLevel(1);
  exatn::TensorNetworkOptimizer optimizer(hubbard_operator,ansatz_full,1e-4);
  success = exatn::sync(); assert(success);
  bool converged = optimizer.optimize();
  success = exatn::sync(); assert(success);
  if(converged){
   std::cout << "Optimization succeeded!" << std::endl;
  }else{
   std::cout << "Optimization failed!" << std::endl; assert(false);
  }
 }
 success = exatn::normalizeNorm2Sync(ansatz_tensor->getName(),1.0); assert(success);
 //success = exatn::printTensor(ansatz_tensor->getName()); assert(success);

 //Reconstruct the exact eigen-tensor as a tensor network:
 ansatz->conjugate();
 success = exatn::balanceNormalizeNorm2Sync(*ansatz_full,1.0,1.0,false); assert(success);
 exatn::TensorNetworkReconstructor::resetDebugLevel(1); //debug
 exatn::TensorNetworkReconstructor reconstructor(ansatz_full,ansatz,1e-7);
 success = exatn::sync(); assert(success);
 double residual_norm, fidelity;
 bool reconstructed = reconstructor.reconstruct(&residual_norm,&fidelity);
 success = exatn::sync(); assert(success);
 if(reconstructed){
  std::cout << "Reconstruction succeeded: Residual norm = " << residual_norm
            << "; Fidelity = " << fidelity << std::endl;
 }else{
  std::cout << "Reconstruction failed!" << std::endl; //assert(false);
 }
 ansatz->conjugate();

 //Perform ground state optimization on a tensor network manifold:
 {
  std::cout << "Ground state optimization on a tensor network manifold:" << std::endl;
  exatn::TensorNetworkOptimizer::resetDebugLevel(1);
  exatn::TensorNetworkOptimizer optimizer(hubbard_operator,ansatz,1e-4);
  //optimizer.resetMicroIterations(1);
  bool converged = optimizer.optimize();
  success = exatn::sync(); assert(success);
  if(converged){
   std::cout << "Optimization succeeded!" << std::endl;
   success = exatn::evaluateSync(*((*ansatz)[0].network)); assert(success);
   success = exatn::normalizeNorm2Sync((*ansatz)[0].network->getTensor(0)->getName(),1.0); assert(success);
   //success = exatn::printTensor((*ansatz)[0].network->getTensor(0)->getName()); assert(success);
  }else{
   std::cout << "Optimization failed!" << std::endl; assert(false);
  }
 }

 //Destroy tensors:
 success = exatn::destroyTensor(ansatz_tensor->getName()); assert(success);
 success = exatn::destroyTensors(*ansatz_net); assert(success);
 for(auto iter = hubbard_operator->begin(); iter != hubbard_operator->end(); ++iter){
  success = exatn::destroyTensorsSync(*(iter->network)); assert(success);
 }

 //Synchronize:
 success = exatn::syncClean(); assert(success);
 exatn::resetLoggingLevel(0,0);
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST28
TEST(NumServerTester, MCVQEHamiltonian) {
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::Tensor;
 using exatn::TensorComposite;
 using exatn::TensorNetwork;
 using exatn::TensorExpansion;
 using exatn::TensorOperator;
 using exatn::TensorElementType;

 const auto TENS_ELEM_TYPE = TensorElementType::COMPLEX64;

 //exatn::resetLoggingLevel(1,2); //debug

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 if(used_mem != 0) exatn::numericalServer->printAllocatedTensors();
 assert(used_mem == 0);

 bool success = true;

 const int num_sites = 8;
 const int bond_dim_lim = 1;
 const int max_bond_dim = std::min(static_cast<int>(std::pow(2,num_sites/2)),bond_dim_lim);
 const int arity = 2;
 const std::string tn_type = "MPS"; //MPS or TTN

 //Read the Hamiltonian in spin representation:
 auto hamiltonian_operator = exatn::quantum::readSpinHamiltonian("MCVQEHam",
     "mcvqe_"+std::to_string(num_sites)+"q.qcw.txt",TENS_ELEM_TYPE,"QCWare");
 success = hamiltonian_operator->deleteComponent(0); assert(success); //remove SCF part
 hamiltonian_operator->printIt();

 //Create tensor network ansatz:
 auto tn_builder = exatn::getTensorNetworkBuilder(tn_type); assert(tn_builder);
 if(tn_type == "MPS"){
  success = tn_builder->setParameter("max_bond_dim",max_bond_dim); assert(success);
 }else if(tn_type == "TTN"){
  success = tn_builder->setParameter("max_bond_dim",max_bond_dim); assert(success);
  success = tn_builder->setParameter("arity",arity); assert(success);
 }else{
  assert(false);
 }
 auto ansatz_tensor = exatn::makeSharedTensor("AnsatzTensor",std::vector<int>(num_sites,2));
 auto ansatz_net = exatn::makeSharedTensorNetwork("Ansatz",ansatz_tensor,*tn_builder);
 ansatz_net->markOptimizableTensors([](const Tensor & tensor){return true;});
 auto ansatz = exatn::makeSharedTensorExpansion("Ansatz",ansatz_net,std::complex<double>{1.0,0.0});
 //ansatz->printIt(); //debug

 //Allocate/initialize tensors in the tensor network ansatz:
 success = exatn::createTensorsSync(*ansatz_net,TENS_ELEM_TYPE); assert(success);
 success = exatn::initTensorsRndSync(*ansatz_net); assert(success);
 /*for(auto tens = ansatz_net->begin(); tens != ansatz_net->end(); ++tens){
  if(tens->first == 0){
   success = exatn::initTensorSync(tens->second.getName(),0.0); assert(success);
  }else{
   success = exatn::initTensorSync(tens->second.getName(),1e-2); assert(success);
  }
 }*/
 //success = exatn::balanceNormalizeNorm2Sync(*ansatz,1.0,1.0,true); assert(success);

 //Perform ground state optimization on a tensor network manifold:
 {
  std::cout << "Ground state optimization on a tensor network manifold:" << std::endl;
  exatn::TensorNetworkOptimizer::resetDebugLevel(1,0);
  exatn::TensorNetworkOptimizer optimizer(hamiltonian_operator,ansatz,1e-4);
  optimizer.enableParallelization(true);
  //optimizer.resetMaxIterations(50);
  //optimizer.resetMicroIterations(1);
  bool converged = optimizer.optimize();
  success = exatn::sync(); assert(success);
  if(converged){
   std::cout << "Optimization succeeded!" << std::endl;
  }else{
   std::cout << "Optimization failed!" << std::endl; assert(false);
  }
 }

 //Destroy tensors:
 success = exatn::destroyTensorsSync(*ansatz_net); assert(success);
 for(auto iter = hamiltonian_operator->begin(); iter != hamiltonian_operator->end(); ++iter){
  success = exatn::destroyTensorsSync(*(iter->network)); assert(success);
 }

 //Synchronize:
 success = exatn::syncClean(); assert(success);
 exatn::resetLoggingLevel(0,0);
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST29
TEST(NumServerTester, TensorOperatorReconstruction) {
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::Tensor;
 using exatn::TensorComposite;
 using exatn::TensorNetwork;
 using exatn::TensorExpansion;
 using exatn::TensorOperator;
 using exatn::TensorElementType;

 const auto TENS_ELEM_TYPE = TensorElementType::COMPLEX64;

 //exatn::resetLoggingLevel(1,2); //debug

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 if(used_mem != 0) exatn::numericalServer->printAllocatedTensors();
 assert(used_mem == 0);

 bool success = true;

 const int num_sites = 8;
 const int bond_dim_lim = 1;
 const int max_bond_dim = std::min(static_cast<int>(std::pow(2,num_sites/2)),bond_dim_lim);
 const int arity = 2;
 const std::string tn_type = "MPS"; //MPS or TTN

 auto tn_builder = exatn::getTensorNetworkBuilder(tn_type); assert(tn_builder);
 if(tn_type == "MPS"){
  success = tn_builder->setParameter("max_bond_dim",max_bond_dim); assert(success);
 }else if(tn_type == "TTN"){
  success = tn_builder->setParameter("max_bond_dim",max_bond_dim); assert(success);
  success = tn_builder->setParameter("arity",arity); assert(success);
 }else{
  assert(false);
 }

 //Build a tensor network operator:
 auto space_tensor = exatn::makeSharedTensor("TensorSpaceMap",std::vector<int>(num_sites*2,2));
 auto ham_net = exatn::makeSharedTensorNetwork("HamiltonianNet",space_tensor,*tn_builder,true);
 ham_net->printIt(); //debug

 //Synchronize:
 success = exatn::syncClean(); assert(success);
 exatn::resetLoggingLevel(0,0);
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST30
TEST(NumServerTester, SpinHamiltonians) {
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::Tensor;
 using exatn::TensorComposite;
 using exatn::TensorNetwork;
 using exatn::TensorExpansion;
 using exatn::TensorOperator;
 using exatn::TensorElementType;
 using exatn::TensorRange;
 using exatn::quantum::Gate;
 using exatn::quantum::PauliMap;
 using exatn::quantum::PauliProduct;

 const auto TENS_ELEM_TYPE = TensorElementType::COMPLEX64;

 const std::complex<double> j_param {-1.0,0.0};
 const std::complex<double> h_param {-0.1,0.0};
 const int num_spin_sites = 4;
 const int bond_dim_lim = 4;
 const int max_bond_dim = std::min(static_cast<int>(std::pow(2,num_spin_sites/2)),bond_dim_lim);
 const int arity = 2;
 const std::string tn_type = "TTN"; //MPS or TTN
 const unsigned int num_states = 4; //only for TTN
 const unsigned int isometric = 1; //only for TTN
 const double accuracy = 1e-4;

 exatn::resetLoggingLevel(1,2); //debug

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

 bool success = true;
 bool root = (exatn::getProcessRank() == 0);

 //Define the 1D Ising Hamiltonian generator:
 TensorRange spin_sites({num_spin_sites});
 auto ising_generator = [j_param,
                         spin_sites,
                         num_sites = spin_sites.localVolume(),
                         finished = false] () mutable -> PauliProduct {
  assert(num_sites > 1);
  PauliProduct pauli_product;
  if(!finished){
   const auto spin_site = spin_sites.localOffset();
   pauli_product.product.emplace_back(PauliMap{Gate::gate_Z,spin_site});
   pauli_product.product.emplace_back(PauliMap{Gate::gate_Z,spin_site+1});
   pauli_product.coefficient = j_param;
   if(spin_site < (num_sites - 2)){
    spin_sites.next();
   }else{
    finished = true;
   }
  }
  return pauli_product;
 };

 //Construct the 1D Ising Hamiltonian using the generator:
 auto ising_hamiltonian0 = exatn::quantum::generateSpinHamiltonian("IsingHamiltonian",
                                                                   ising_generator,
                                                                   TENS_ELEM_TYPE);
 //ising_hamiltonian0->printIt(); //debug

 //Define the 1D transverse field generator:
 spin_sites.reset();
 auto transverse_generator = [h_param,
                              spin_sites,
                              num_sites = spin_sites.localVolume(),
                              finished = false] () mutable -> PauliProduct {
  assert(num_sites > 1);
  PauliProduct pauli_product;
  if(!finished){
   const auto spin_site = spin_sites.localOffset();
   pauli_product.product.emplace_back(PauliMap{Gate::gate_X,spin_site});
   pauli_product.coefficient = h_param;
   if(spin_site < (num_sites - 1)){
    spin_sites.next();
   }else{
    finished = true;
   }
  }
  return pauli_product;
 };

 //Construct the 1D transverse field Hamiltonian using the generator:
 auto transverse_field = exatn::quantum::generateSpinHamiltonian("TransverseField",
                                                                 transverse_generator,
                                                                 TENS_ELEM_TYPE);
 //transverse_field->printIt(); //debug

 //Construct the full transverse field Ising Hamiltonian:
 auto transverse_ising = exatn::combineTensorOperators(*ising_hamiltonian0,*transverse_field);
 //transverse_ising->printIt(); //debug

 //Configure the tensor network builder:
 auto tn_builder = exatn::getTensorNetworkBuilder(tn_type); assert(tn_builder);
 if(tn_type == "MPS"){
  success = tn_builder->setParameter("max_bond_dim",max_bond_dim); assert(success);
 }else if(tn_type == "TTN"){
  success = tn_builder->setParameter("max_bond_dim",max_bond_dim); assert(success);
  success = tn_builder->setParameter("arity",arity); assert(success);
  success = tn_builder->setParameter("isometric",isometric); assert(success);
  if(isometric != 0){
   success = tn_builder->setParameter("num_states",num_states); assert(success);
  }
 }else{
  assert(false);
 }

 //Build tensor network vectors:
 auto ket_tensor = exatn::makeSharedTensor("TensorSpace",std::vector<int>(num_spin_sites,2));
 auto vec_net0 = exatn::makeSharedTensorNetwork("VectorNet0",ket_tensor,*tn_builder,false);
 vec_net0->getTensor(3)->replaceDimension(2,2); //debug
 vec_net0->getTensor(0)->replaceDimension(4,2); //debug
 vec_net0->markOptimizableAllTensors();
 auto vec_tns0 = exatn::makeSharedTensorExpansion("VectorTNS0",vec_net0,std::complex<double>{1.0,0.0});
 auto rhs_net = exatn::makeSharedTensorNetwork("RightHandSideNet",ket_tensor,*tn_builder,false);
 auto rhs_tns = exatn::makeSharedTensorExpansion("RightHandSideTNS",rhs_net,std::complex<double>{1.0,0.0});
 vec_net0->printIt(); //debug

 //Numerical processing:
 {
  //Create and initialize tensor network vector tensors:
  if(root) std::cout << "Creating and initializing tensor network vector tensors ... ";
  success = exatn::createTensorsSync(*vec_net0,TENS_ELEM_TYPE); assert(success);
  success = exatn::initTensorsRndSync(*vec_net0); assert(success);
  success = exatn::createTensorsSync(*rhs_net,TENS_ELEM_TYPE); assert(success);
  success = exatn::initTensorsRndSync(*rhs_net); assert(success);
  if(root) std::cout << "Ok" << std::endl;

  //Ground and excited states in one call:
  if(root) std::cout << "Ground and excited states search for the original Hamiltonian:" << std::endl;
  exatn::TensorNetworkOptimizer::resetDebugLevel(1,0);
  exatn::TensorNetworkOptimizer optimizer(transverse_ising,vec_tns0,accuracy);
  optimizer.enableParallelization(true);
  success = exatn::sync(); assert(success);
  bool converged = false;
  if(isometric != 0){
   converged = optimizer.optimize(); //state-averaged
  }else{
   converged = optimizer.optimizeSequential(num_states); //state-specific
  }
  success = exatn::sync(); assert(success);
  if(converged){
   if(root){
    std::cout << "Search succeeded:" << std::endl;
    if(isometric != 0){
     std::cout << "State-averaged expectation value for " << num_states << " states = "
               << optimizer.getExpectationValue() << std::endl;
    }else{
     for(unsigned int root_id = 0; root_id < num_states; ++root_id){
      std::cout << "Expectation value " << root_id << " = "
                << optimizer.getExpectationValue(root_id) << std::endl;
     }
    }
   }
  }else{
   if(root) std::cout << "Search failed!" << std::endl;
   assert(false);
  }
 }

 //Synchronize:
 success = exatn::destroyTensorsSync(); assert(success);
 success = exatn::syncClean(); assert(success);
 exatn::resetLoggingLevel(0,0);
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST31
TEST(NumServerTester, ExcitedMCVQE) {
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::Tensor;
 using exatn::TensorComposite;
 using exatn::TensorNetwork;
 using exatn::TensorExpansion;
 using exatn::TensorOperator;
 using exatn::TensorElementType;
 using exatn::TensorRange;
 using exatn::quantum::Gate;
 using exatn::quantum::PauliMap;
 using exatn::quantum::PauliProduct;

 const auto TENS_ELEM_TYPE = TensorElementType::COMPLEX64;

 const int num_spin_sites = 8;
 const int bond_dim_lim = 16;
 const int max_bond_dim = std::min(static_cast<int>(std::pow(2,num_spin_sites/2)),bond_dim_lim);
 const int arity = 2;
 const std::string tn_type = "TTN"; //MPS or TTN
 const unsigned int num_states = 4;
 const double accuracy = 3e-5;

 //exatn::resetLoggingLevel(1,2); //debug

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

 bool success = true;
 bool root = (exatn::getProcessRank() == 0);

 //Read the MCVQE Hamiltonian in spin representation:
 auto hamiltonian0 = exatn::quantum::readSpinHamiltonian("MCVQEHamiltonian",
  "mcvqe_"+std::to_string(num_spin_sites)+"q.qcw.txt",TENS_ELEM_TYPE,"QCWare");
 success = hamiltonian0->deleteComponent(0); assert(success);

 //Configure the tensor network builder:
 auto tn_builder = exatn::getTensorNetworkBuilder(tn_type); assert(tn_builder);
 if(tn_type == "MPS"){
  success = tn_builder->setParameter("max_bond_dim",max_bond_dim); assert(success);
 }else if(tn_type == "TTN"){
  success = tn_builder->setParameter("max_bond_dim",max_bond_dim); assert(success);
  success = tn_builder->setParameter("arity",arity); assert(success);
 }else{
  assert(false);
 }

 //Build tensor network vectors:
 auto ket_tensor = exatn::makeSharedTensor("TensorSpace",std::vector<int>(num_spin_sites,2));
 auto vec_net0 = exatn::makeSharedTensorNetwork("VectorNet1",ket_tensor,*tn_builder,false);
 vec_net0->markOptimizableAllTensors();
 auto vec_tns0 = exatn::makeSharedTensorExpansion("VectorTNS1",vec_net0,std::complex<double>{1.0,0.0});
 auto vec_net1 = exatn::makeSharedTensorNetwork("VectorNet2",ket_tensor,*tn_builder,false);
 vec_net1->markOptimizableAllTensors();
 auto vec_tns1 = exatn::makeSharedTensorExpansion("VectorTNS2",vec_net1,std::complex<double>{1.0,0.0});
 auto vec_net2 = exatn::makeSharedTensorNetwork("VectorNet3",ket_tensor,*tn_builder,false);
 vec_net2->markOptimizableAllTensors();
 auto vec_tns2 = exatn::makeSharedTensorExpansion("VectorTNS3",vec_net2,std::complex<double>{1.0,0.0});

 //Numerical processing:
 {
  exatn::switchComputationalBackend("default");
  //Create and initialize tensor network vector tensors:
  if(root) std::cout << "Creating and initializing tensor network vector tensors ... ";
  success = exatn::createTensorsSync(*vec_net0,TENS_ELEM_TYPE); assert(success);
  success = exatn::initTensorsRndSync(*vec_net0); assert(success);
  success = exatn::createTensorsSync(*vec_net1,TENS_ELEM_TYPE); assert(success);
  success = exatn::initTensorsRndSync(*vec_net1); assert(success);
  success = exatn::createTensorsSync(*vec_net2,TENS_ELEM_TYPE); assert(success);
  success = exatn::initTensorsRndSync(*vec_net2); assert(success);
  if(root) std::cout << "Ok" << std::endl;
#if 0
  //Ground state search for the original Hamiltonian:
  if(root) std::cout << "Ground state search for the original Hamiltonian:" << std::endl;
  exatn::TensorNetworkOptimizer::resetDebugLevel(1,0);
  exatn::TensorNetworkOptimizer optimizer0(hamiltonian0,vec_tns0,accuracy);
  success = exatn::sync(); assert(success);
  bool converged = optimizer0.optimize();
  success = exatn::sync(); assert(success);
  if(converged){
   if(root) std::cout << "Search succeeded: ";
  }else{
   if(root) std::cout << "Search failed!" << std::endl;
   assert(false);
  }
  const auto expect_val0 = optimizer0.getExpectationValue();
  if(root) std::cout << "Expectation value = " << expect_val0 << std::endl;

  //First excited state search for the projected Hamiltonian:
  if(root) std::cout << "1st excited state search for the projected Hamiltonian:" << std::endl;
  vec_net0->markOptimizableNoTensors();
  std::vector<std::pair<unsigned int, unsigned int>> ket_pairing(num_spin_sites);
  for(unsigned int i = 0; i < num_spin_sites; ++i) ket_pairing[i] = std::make_pair(i,i);
  std::vector<std::pair<unsigned int, unsigned int>> bra_pairing(num_spin_sites);
  for(unsigned int i = 0; i < num_spin_sites; ++i) bra_pairing[i] = std::make_pair(i,i);
  auto projector0 = exatn::makeSharedTensorOperator("Projector0",vec_net0,vec_net0,
                                                    ket_pairing,bra_pairing,-expect_val0);
  auto hamiltonian1 = exatn::combineTensorOperators(*hamiltonian0,*projector0);
  exatn::TensorNetworkOptimizer::resetDebugLevel(1,0);
  exatn::TensorNetworkOptimizer optimizer1(hamiltonian1,vec_tns1,accuracy);
  success = exatn::sync(); assert(success);
  converged = optimizer1.optimize();
  success = exatn::sync(); assert(success);
  if(converged){
   if(root) std::cout << "Search succeeded: ";
  }else{
   if(root) std::cout << "Search failed!" << std::endl;
   assert(false);
  }
  const auto expect_val1 = optimizer1.getExpectationValue();
  if(root) std::cout << "Expectation value = " << expect_val1 << std::endl;

  //Second excited state search for the projected Hamiltonian:
  if(root) std::cout << "2nd excited state search for the projected Hamiltonian:" << std::endl;
  vec_net1->markOptimizableNoTensors();
  auto projector1 = exatn::makeSharedTensorOperator("Projector1",vec_net1,vec_net1,
                                                    ket_pairing,bra_pairing,-expect_val1);
  auto hamiltonian2 = exatn::combineTensorOperators(*hamiltonian1,*projector1);
  exatn::TensorNetworkOptimizer::resetDebugLevel(1,0);
  exatn::TensorNetworkOptimizer optimizer2(hamiltonian2,vec_tns2,accuracy);
  success = exatn::sync(); assert(success);
  converged = optimizer2.optimize();
  success = exatn::sync(); assert(success);
  if(converged){
   if(root) std::cout << "Search succeeded: ";
  }else{
   if(root) std::cout << "Search failed!" << std::endl;
   assert(false);
  }
  const auto expect_val2 = optimizer2.getExpectationValue();
  if(root) std::cout << "Expectation value = " << expect_val2 << std::endl;
#endif
  //Ground and three excited states in one call:
  if(root) std::cout << "Ground and excited states search for the original Hamiltonian:" << std::endl;
  exatn::TensorNetworkOptimizer::resetDebugLevel(1,0);
  vec_net0->markOptimizableAllTensors();
  success = exatn::initTensorsRndSync(*vec_tns0); assert(success);
  exatn::TensorNetworkOptimizer optimizer3(hamiltonian0,vec_tns0,accuracy);
  optimizer3.enableParallelization(true);
  success = exatn::sync(); assert(success);
  bool converged = optimizer3.optimizeSequential(num_states);
  success = exatn::sync(); assert(success);
  if(converged){
   if(root){
    std::cout << "Search succeeded:" << std::endl;
    for(unsigned int root_id = 0; root_id < num_states; ++root_id){
     std::cout << "Expectation value " << root_id << " = "
               << optimizer3.getExpectationValue(root_id) << std::endl;
    }
   }
  }else{
   if(root) std::cout << "Search failed!" << std::endl;
   assert(false);
  }
 }

 //Synchronize:
 success = exatn::destroyTensorsSync(); assert(success);
 success = exatn::syncClean(); assert(success);
 exatn::resetLoggingLevel(0,0);
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST32
TEST(NumServerTester, IsometricAIEM) {
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::Tensor;
 using exatn::TensorNetwork;
 using exatn::TensorExpansion;
 using exatn::TensorOperator;
 using exatn::TensorElementType;
 using exatn::TensorRange;

 const auto TENS_ELEM_TYPE = TensorElementType::COMPLEX64;

 const int num_spin_sites = 8;
 const int bond_dim_lim = 16;
 const int max_bond_dim = std::min(static_cast<int>(std::pow(2,num_spin_sites/2)),bond_dim_lim);
 const std::string tn_type = "TTN"; //MPS or TTN
 const int arity = 2;
 const unsigned int isometric = 1;
 const unsigned int num_states = 4;
 const bool multistate = (num_states > 1 && isometric != 0);
 const unsigned int max_iterations = 1000;
 const double accuracy = 3e-5;

 //exatn::resetLoggingLevel(1,2); //debug

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

 bool success = true;
 bool root = (exatn::getProcessRank() == 0);

 //Create tensors:
 success = exatn::createTensor("A",TENS_ELEM_TYPE,TensorShape{4,2,8,2,4}); assert(success);
 exatn::getTensor("A")->registerIsometry({0,2,4});
 success = exatn::createTensor("B",TENS_ELEM_TYPE,TensorShape{2,2,2,2}); assert(success);

 //Init tensors:
 success = exatn::initTensorRnd("A"); assert(success); //random initialization will respect isometry
 //success = exatn::transformTensor("A",std::shared_ptr<exatn::TensorMethod>(
 //           new exatn::FunctorIsometrize(std::vector<unsigned int>{0,2,4})));
 //assert(success);
 success = exatn::initTensor("B",0.0); assert(success);

 //Contract tensors (to produce identity):
 success = exatn::contractTensorsSync("B(j1,j2,j3,j4)+=A(i1,j1,i2,j2,i3)*A+(i1,j3,i2,j4,i3)",1.0); assert(success);
 //success = exatn::printTensor("B"); assert(success);
 double nrm1 = 0.0;
 success = exatn::computeNorm1Sync("B",nrm1); assert(success);
 if(root) std::cout << "1-norm of the identity tensor = " << nrm1 << " VS correct = 4" << std::endl;
 exatn::make_sure(nrm1,4.0,1e-6);

 //Destroy tensors:
 success = exatn::sync(); assert(success);
 success = exatn::destroyTensor("B"); assert(success);
 success = exatn::destroyTensor("A"); assert(success);

 //Read the MCVQE Hamiltonian in spin representation:
 if(root) std::cout << "Reading in the Hamiltonian ... ";
 auto hamiltonian = exatn::quantum::readSpinHamiltonian("MCVQEHamiltonian",
  "mcvqe_"+std::to_string(num_spin_sites)+"q.qcw.txt",TENS_ELEM_TYPE,"QCWare");
 success = hamiltonian->deleteComponent(0); assert(success);
 if(root) std::cout << "Done" << std::endl;

 if(root) std::cout << "Constructing the tensor network ansatz ... ";
 //Configure the tensor network builder:
 auto tn_builder = exatn::getTensorNetworkBuilder(tn_type); assert(tn_builder);
 if(tn_type == "MPS"){
  success = tn_builder->setParameter("max_bond_dim",max_bond_dim); assert(success);
 }else if(tn_type == "TTN"){
  success = tn_builder->setParameter("max_bond_dim",max_bond_dim); assert(success);
  success = tn_builder->setParameter("arity",arity); assert(success);
  success = tn_builder->setParameter("num_states",num_states); assert(success);
  success = tn_builder->setParameter("isometric",isometric); assert(success);
 }else{
  std::cout << "#FATAL: Unknown tensor network builder!" << std::endl; assert(false);
 }

 //Construct the tensor network ansatz:
 auto ket_tensor = exatn::makeSharedTensor("TensorSpace",std::vector<int>(num_spin_sites,2));
 auto ket_net = exatn::makeSharedTensorNetwork("KetNet",ket_tensor,*tn_builder,false);
 auto ket_tns = exatn::makeSharedTensorExpansion("KetTNS",ket_net,std::complex<double>{1.0,0.0});
 if(root) std::cout << "Done" << std::endl;
 if(root) ket_tns->printIt(); //debug

 //Checking the Hamiltonian functional structure:
 if(root) std::cout << "Checking the Hamiltonian functional structure:\n";
 if(root) std::cout << " Conjugated (bra) TNS:\n";
 auto bra_tns = exatn::makeSharedTensorExpansion(*ket_tns);
 bra_tns->conjugate();
 bra_tns->rename("BraTNS");
 if(root) bra_tns->printIt(); //debug
 if(root) std::cout << " Full Hamiltonian functional:\n";
 auto func_tns = exatn::makeSharedTensorExpansion(*bra_tns,*ket_tns,*hamiltonian);
 if(root) func_tns->printIt(); //debug
 if(root) std::cout << " Collapsed Hamiltonian functional: ";
 bool deltas = false;
 auto collapsed = func_tns->collapseIsometries(&deltas);
 if(root) std::cout << "Collapsed = " << collapsed << "; Deltas appended = " << deltas << std::endl;
 if(root) func_tns->printIt(); //debug
 if(root) std::cout << "Done" << std::endl;

 {//Numerical processing:
  if(root) std::cout << "Allocating and initializing the tensor network ansatz ... ";
  success = exatn::createTensorsSync(*ket_net,TENS_ELEM_TYPE); assert(success);
  success = exatn::initTensorsRndSync(*ket_tns); assert(success);
  if(root) std::cout << "Done" << std::endl;

  if(root) std::cout << "Ground and excited states search for the original Hamiltonian:" << std::endl;
  exatn::TensorNetworkOptimizer::resetDebugLevel(1,0);
  success = exatn::initTensorsWithIsometriesSync(*ket_net); assert(success);
  ket_net->markOptimizableAllTensors();
  //ket_net->markOptimizableTensor(7); //debug
  exatn::TensorNetworkOptimizer optimizer(hamiltonian,ket_tns,accuracy);
  optimizer.enableParallelization(true);
  optimizer.resetMaxIterations(max_iterations);
  success = exatn::sync(); assert(success);
  bool converged = optimizer.optimize(multistate); //single- or multi-state tensor network optimized in one shot
  //bool converged = optimizer.optimizeSequential(num_states); //sequential state-by-state optimization using a single-state tensor network
  success = exatn::sync(); assert(success);
  if(converged){
   if(root){
    std::cout << "Search succeeded:" << std::endl;
    /*for(unsigned int root_id = 0; root_id < num_states; ++root_id){
     std::cout << "Expectation value " << root_id << " = "
               << optimizer.getExpectationValue(root_id) << std::endl;
    }*/
   }
  }else{
   if(root) std::cout << "Search failed!" << std::endl;
   assert(false);
  }
 }

 //Synchronize:
 if(root) std::cout << "Destroying all tensors ... ";
 success = exatn::destroyTensorsSync(); assert(success);
 if(root) std::cout << "Done" << std::endl;
 success = exatn::syncClean(); assert(success);
 exatn::resetLoggingLevel(0,0);
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST33
TEST(NumServerTester, CuTensorNet) {
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::Tensor;
 using exatn::TensorNetwork;
 using exatn::TensorExpansion;
 using exatn::TensorOperator;
 using exatn::TensorElementType;
 using exatn::TensorRange;

 const auto TENS_ELEM_TYPE = TensorElementType::REAL32;
 const std::size_t DIM_EXT = 64; //64 is default
 const int NUM_REPEATS = 10;

 //exatn::resetLoggingLevel(1,1); //debug

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

 //exatn::resetContrSeqOptimizer("metis",false,true);
 //exatn::resetContrSeqOptimizer("cutnn",false,true);
 //exatn::resetContrSeqOptimizer("cutnn",false,false);

 bool success = true;

 //Create tensors:
 success = exatn::createTensor("A",TENS_ELEM_TYPE,TensorShape{DIM_EXT,DIM_EXT,DIM_EXT,DIM_EXT}); assert(success);
 success = exatn::createTensor("B",TENS_ELEM_TYPE,TensorShape{DIM_EXT,DIM_EXT,DIM_EXT}); assert(success);
 success = exatn::createTensor("C",TENS_ELEM_TYPE,TensorShape{DIM_EXT,DIM_EXT,DIM_EXT}); assert(success);
 success = exatn::createTensor("D",TENS_ELEM_TYPE,TensorShape{DIM_EXT,DIM_EXT,DIM_EXT,DIM_EXT}); assert(success);
 success = exatn::createTensor("E",TENS_ELEM_TYPE,TensorShape{DIM_EXT,DIM_EXT,DIM_EXT,DIM_EXT}); assert(success);
 success = exatn::createTensor("F",TENS_ELEM_TYPE,TensorShape{DIM_EXT,DIM_EXT,DIM_EXT,DIM_EXT}); assert(success);
 success = exatn::createTensor("G",TENS_ELEM_TYPE,TensorShape{DIM_EXT,DIM_EXT,DIM_EXT,DIM_EXT}); assert(success);

 //Init tensors:
 success = exatn::initTensorRnd("A"); assert(success);
 success = exatn::initTensorRnd("B"); assert(success);
 success = exatn::initTensorRnd("C"); assert(success);
 success = exatn::initTensor("D",0.0); assert(success);
 success = exatn::initTensor("E",0.0); assert(success);
 success = exatn::initTensor("F",0.0); assert(success);
 success = exatn::initTensor("G",0.0); assert(success);

 success = exatn::sync(); assert(success);

 std::cout << "Testing individual tensor network execution via default backend ...\n";
 int num_repeats = NUM_REPEATS;
 while(--num_repeats >= 0){
  std::cout << "D(m,x,n,y)+=A(m,h,k,n)*B(u,k,h)*C(x,u,y): ";
  auto flops = exatn::getTotalFlopCount();
  auto time_start = exatn::Timer::timeInSecHR();
  success = exatn::evaluateTensorNetwork("cuNet","D(m,x,n,y)+=A(m,h,k,n)*B(u,k,h)*C(x,u,y)"); assert(success);
  success = exatn::sync("D",true); assert(success);
  auto duration = exatn::Timer::timeInSecHR(time_start);
  flops = exatn::getTotalFlopCount() - flops;
  std::cout << "Duration = " << duration << " s; GFlop count = " << flops/1e9
            << "; Performance = " << (flops / (1e9 * duration)) << " Gflop/s\n";
  double norm = 0.0;
  success = exatn::computeNorm1Sync("D",norm); assert(success);
  std::cout << "1-norm of tensor D = " << norm << std::endl;
 }

#ifdef CUQUANTUM
 success = exatn::sync(); assert(success);
 auto backends = exatn::queryComputationalBackends();
 if(std::find(backends.cbegin(),backends.cend(),"cuquantum") != backends.cend())
  exatn::switchComputationalBackend("cuquantum");

 std::cout << "Testing individual tensor network execution via cuQuantum backend ...\n";
 num_repeats = NUM_REPEATS;
 while(--num_repeats >= 0){
  std::cout << "D(m,x,n,y)+=A(m,h,k,n)*B(u,k,h)*C(x,u,y): ";
  auto flops = exatn::getTotalFlopCount();
  auto time_start = exatn::Timer::timeInSecHR();
  success = exatn::evaluateTensorNetwork("cuNet","D(m,x,n,y)+=A(m,h,k,n)*B(u,k,h)*C(x,u,y)"); assert(success);
  success = exatn::sync("D",true); assert(success);
  auto duration = exatn::Timer::timeInSecHR(time_start);
  flops = exatn::getTotalFlopCount() - flops;
  std::cout << "Duration = " << duration << " s; GFlop count = " << flops/1e9
            << "; Performance = " << (flops / (1e9 * duration)) << " Gflop/s\n";
  double norm = 0.0;
  success = exatn::computeNorm1Sync("D",norm); assert(success);
  std::cout << "1-norm of tensor D = " << norm << std::endl;
 }

 success = exatn::sync(); assert(success);

 std::cout << "Testing tensor network execution pipelining ...\n";
 num_repeats = NUM_REPEATS;
 while(--num_repeats >= 0){
  auto flops = exatn::getTotalFlopCount();
  auto time_start = exatn::Timer::timeInSecHR();
  std::cout << "D(m,x,n,y)+=A(m,h,k,n)*B(u,k,h)*C(x,u,y): ";
  success = exatn::evaluateTensorNetwork("cuNet","D(m,x,n,y)+=A(m,h,k,n)*B(u,k,h)*C(x,u,y)"); assert(success);
  std::cout << "E(m,x,n,y)+=A(m,h,k,n)*B(u,k,h)*C(x,u,y): ";
  success = exatn::evaluateTensorNetwork("cuNet","E(m,x,n,y)+=A(m,h,k,n)*B(u,k,h)*C(x,u,y)"); assert(success);
  std::cout << "F(m,x,n,y)+=A(m,h,k,n)*B(u,k,h)*C(x,u,y): ";
  success = exatn::evaluateTensorNetwork("cuNet","F(m,x,n,y)+=A(m,h,k,n)*B(u,k,h)*C(x,u,y)"); assert(success);
  std::cout << "G(m,x,n,y)+=A(m,h,k,n)*B(u,k,h)*C(x,u,y): ";
  success = exatn::evaluateTensorNetwork("cuNet","G(m,x,n,y)+=A(m,h,k,n)*B(u,k,h)*C(x,u,y)"); assert(success);
  success = exatn::sync(); assert(success);
  auto duration = exatn::Timer::timeInSecHR(time_start);
  flops = exatn::getTotalFlopCount() - flops;
  std::cout << "Duration = " << duration << " s; GFlop count = " << flops/1e9
            << "; Performance = " << (flops / (1e9 * duration)) << " Gflop/s\n";
  double norm = 0.0;
  success = exatn::computeNorm1Sync("D",norm); assert(success);
  std::cout << "1-norm of tensor D = " << norm << std::endl;
  norm = 0.0;
  success = exatn::computeNorm1Sync("E",norm); assert(success);
  std::cout << "1-norm of tensor E = " << norm << std::endl;
  norm = 0.0;
  success = exatn::computeNorm1Sync("F",norm); assert(success);
  std::cout << "1-norm of tensor F = " << norm << std::endl;
  norm = 0.0;
  success = exatn::computeNorm1Sync("G",norm); assert(success);
  std::cout << "1-norm of tensor G = " << norm << std::endl;
 }
#endif

 //Destroy tensors:
 success = exatn::sync(); assert(success);
 success = exatn::destroyTensor("G"); assert(success);
 success = exatn::destroyTensor("F"); assert(success);
 success = exatn::destroyTensor("E"); assert(success);
 success = exatn::destroyTensor("D"); assert(success);
 success = exatn::destroyTensor("C"); assert(success);
 success = exatn::destroyTensor("B"); assert(success);
 success = exatn::destroyTensor("A"); assert(success);

 //Synchronize:
 success = exatn::syncClean(); assert(success);
 exatn::resetLoggingLevel(0,0);
 //Grab a beer!
}
#endif

#ifdef EXATN_TEST34
TEST(NumServerTester, TensorComposite) {
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::Tensor;
 using exatn::TensorComposite;
 using exatn::TensorNetwork;
 using exatn::TensorExpansion;
 using exatn::TensorOperator;
 using exatn::TensorElementType;

 const auto TENS_ELEM_TYPE = TensorElementType::COMPLEX32;

 //exatn::resetLoggingLevel(1,2); //debug

 std::size_t free_mem = 0;
 auto used_mem = exatn::getMemoryUsage(&free_mem);
 std::cout << "#MSG(exatn): Backend tensor memory usage on entrance = "
           << used_mem << std::endl << std::flush;
 assert(used_mem == 0);

 bool success = true;

 // Determine parallel configuration:
 const auto & all_processes = exatn::getDefaultProcessGroup();
 const auto my_process_rank = exatn::getProcessRank(all_processes);
 const auto total_ranks = exatn::getNumProcesses(all_processes);
 std::cout << "Process " << my_process_rank << " (total number of MPI processes = "
           << total_ranks << ")" << std::endl;

 //Create composite tensors:
 success = exatn::createTensorSync(all_processes,"A",
                                   std::vector<std::pair<unsigned int, unsigned int>>{{1,2},{0,1}},
                                   TENS_ELEM_TYPE,TensorShape{100,60}); assert(success);
 auto tensorA = exatn::castTensorComposite(exatn::getTensor("A")); assert(tensorA);

 /*auto num_subtensors = tensorA->getNumSubtensors();
 if(my_process_rank == 0){
  for(unsigned long long i = 0; i < num_subtensors; ++i){
   std::cout << "Process " << my_process_rank << ": Subtensor " << i << ": Closest owner process is "
             << exatn::subtensor_owner_id(my_process_rank,total_ranks,i,num_subtensors)
             << ": "; (*tensorA)[i]->printIt(); std::cout << std::endl;
  }
 }

 auto owned = exatn::owned_subtensors(my_process_rank,total_ranks,num_subtensors);
 std::cout << "Process " << my_process_rank << " owns subtensors ["
           << owned.first << ".." << owned.second << "]" << std::endl;*/

 success = exatn::createTensorSync(all_processes,"B",
                                   std::vector<std::pair<unsigned int, unsigned int>>{{0,1},{1,1}},
                                   TENS_ELEM_TYPE,TensorShape{100,60}); assert(success);
 auto tensorB = exatn::castTensorComposite(exatn::getTensor("B")); assert(tensorB);

 /*num_subtensors = tensorB->getNumSubtensors();
 if(my_process_rank == 0){
  for(unsigned long long i = 0; i < num_subtensors; ++i){
   std::cout << "Process " << my_process_rank << ": Subtensor " << i << ": Closest owner process is "
             << exatn::subtensor_owner_id(my_process_rank,total_ranks,i,num_subtensors)
             << ": "; (*tensorB)[i]->printIt(); std::cout << std::endl;
  }
 }

 owned = exatn::owned_subtensors(my_process_rank,total_ranks,num_subtensors);
 std::cout << "Process " << my_process_rank << " owns subtensors ["
           << owned.first << ".." << owned.second << "]" << std::endl;*/

 success = exatn::createTensorSync(all_processes,"C",
                                   std::vector<std::pair<unsigned int, unsigned int>>{{1,1},{0,1}},
                                   TENS_ELEM_TYPE,TensorShape{60,60}); assert(success);
 auto tensorC = exatn::castTensorComposite(exatn::getTensor("C")); assert(tensorC);

 //std::cout << "Process " << my_process_rank << " is within existence domain of {A,B,C} = "
 //          << exatn::withinTensorExistenceDomain("A","B","C") << std::endl; //debug

 //Initialize composite tensors:
 success = exatn::initTensor("A",std::complex<float>{1e-3,0.0}); assert(success);
 success = exatn::initTensor("B",std::complex<float>{0.0,0.0}); assert(success);
 success = exatn::initTensor("C",std::complex<float>{1e-4,0.0}); assert(success);

 //Sync all processes:
 success = exatn::sync(); assert(success);

 //Compute the 1-norm of a composite tensor:
 double norm = 0.0;
 success = exatn::computeNorm1Sync("C",norm); assert(success);
 std::cout << "1-norm of tensor C = " << norm << std::endl;
 //Compute the 2-norm of a composite tensor:
 norm = 0.0;
 success = exatn::computeNorm2Sync("C",norm); assert(success);
 std::cout << "2-norm of tensor C = " << norm << std::endl;
 //Compute the Max-Abs of a composite tensor:
 norm = 0.0;
 success = exatn::computeMaxAbsSync("C",norm); assert(success);
 std::cout << "Max-Abs of tensor C = " << norm << std::endl;
 //Compute partial norms of a composite tensor:
 norm = 0.0;
 success = exatn::computeNorm2Sync("A",norm); assert(success);
 std::vector<double> pnorms;
 success = exatn::computePartialNormsSync("A",1,pnorms); assert(success);
 auto pnorms_sum = std::accumulate(pnorms.cbegin(),pnorms.cend(),0.0);
 std::cout << "Sum of partial norms of tensor A = " << pnorms_sum
           << " VS full norm = " << (norm * norm) << std::endl;

 //Copy a composite tensor:
 success = exatn::addTensorsSync("B(i,j)+=A(i,j)",1.0); assert(success);
 norm = 0.0;
 success = exatn::computeNorm2Sync("B",norm); assert(success);
 std::cout << "2-norm of tensor B = " << (norm * norm) << std::endl;
#if 0
 //Contract composite tensors:
 success = exatn::contractTensorsSync("C(i,j)+=A(k,i)*B(k,j)",1.0); assert(success);
 norm = 0.0;
 success = exatn::computeNorm1Sync("C",norm); assert(success);
 std::cout << "1-norm of tensor C after contraction = " << norm << std::endl;
#endif
 //Destroy composite tensors:
 success = exatn::sync(); assert(success);
 success = exatn::destroyTensorSync("C"); assert(success);
 success = exatn::destroyTensorSync("B"); assert(success);
 success = exatn::destroyTensorSync("A"); assert(success);

 //Synchronize:
 success = exatn::destroyTensorsSync(); assert(success);
 success = exatn::syncClean(); assert(success);
 exatn::resetLoggingLevel(0,0);
 //Grab a beer!
}
#endif


int main(int argc, char **argv) {

  exatn::ParamConf exatn_parameters;
  //Set the available CPU Host RAM size to be used by ExaTN:
  exatn_parameters.setParameter("host_memory_buffer_size",4L*1024L*1024L*1024L);
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

  bool success = exatn::syncClean(); assert(success);
  std::size_t free_mem = 0;
  std::size_t used_mem = exatn::getMemoryUsage(&free_mem);
  std::cout << "Global backend tensor memory usage on exit = "
            << used_mem << std::endl << std::flush;
  assert(used_mem == 0);

  exatn::finalize();
#ifdef MPI_ENABLED
  mpi_error = MPI_Finalize(); assert(mpi_error == MPI_SUCCESS);
#endif
  return ret;
}
