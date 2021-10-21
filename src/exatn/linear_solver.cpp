/** ExaTN:: Linear solver over tensor network manifolds
REVISION: 2021/10/21

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "linear_solver.hpp"
#include "reconstructor.hpp"

namespace exatn{

unsigned int TensorNetworkLinearSolver::debug{0};
int TensorNetworkLinearSolver::focus{-1};


TensorNetworkLinearSolver::TensorNetworkLinearSolver(std::shared_ptr<TensorOperator> tensor_operator,
                                                     std::shared_ptr<TensorExpansion> rhs_expansion,
                                                     std::shared_ptr<TensorExpansion> vector_expansion,
                                                     double tolerance):
 tensor_operator_(tensor_operator), rhs_expansion_(rhs_expansion), vector_expansion_(vector_expansion),
 max_iterations_(DEFAULT_MAX_ITERATIONS), tolerance_(tolerance),
#ifdef MPI_ENABLED
 parallel_(true)
#else
 parallel_(false)
#endif
{
 if(!rhs_expansion_->isKet()){
  std::cout << "#ERROR(exatn:TensorNetworkLinearSolver): The rhs tensor network vector expansion must be a ket!"
            << std::endl << std::flush;
  assert(false);
 }
 if(!vector_expansion_->isKet()){
  std::cout << "#ERROR(exatn:TensorNetworkLinearSolver): The tensor network vector expansion sought for must be a ket!"
            << std::endl << std::flush;
  assert(false);
 }
}


void TensorNetworkLinearSolver::resetTolerance(double tolerance)
{
 tolerance_ = tolerance;
 return;
}


void TensorNetworkLinearSolver::resetMaxIterations(unsigned int max_iterations)
{
 max_iterations_ = max_iterations;
 return;
}


std::shared_ptr<TensorExpansion> TensorNetworkLinearSolver::getSolution() const
{
 return vector_expansion_;
}


bool TensorNetworkLinearSolver::solve()
{
 return solve(exatn::getDefaultProcessGroup());
}


bool TensorNetworkLinearSolver::solve(const ProcessGroup & process_group)
{
 unsigned int local_rank; //local process rank within the process group
 if(!process_group.rankIsIn(exatn::getProcessRank(),&local_rank)) return true; //process is not in the group: Do nothing
 unsigned int num_procs = 1;
 if(parallel_) num_procs = process_group.getSize();

 if(TensorNetworkLinearSolver::focus >= 0){
  if(getProcessRank() != TensorNetworkLinearSolver::focus) TensorNetworkLinearSolver::debug = 0;
 }

 //Construct the |A*|b> tensor network expansion:
 rhs_expansion_->conjugate();
 opvec_expansion_ = std::make_shared<TensorExpansion>(*rhs_expansion_,*tensor_operator_);
 opvec_expansion_->conjugate();
 rhs_expansion_->conjugate();

 //Normalize |A*|b> to unity (<b|A|A*|b> = 1):
 double original_norm = 0.0;
 bool success = normalizeNorm2Sync(process_group,*opvec_expansion_,1.0,&original_norm); assert(success);
 if(TensorNetworkLinearSolver::debug > 0)
  std::cout << "#DEBUG(exatn::TensorNetworkLinearSolver): Original <b|A| norm = " << original_norm << std::endl;

 //Solve for <x| via reconstruction (<x||A*|b>):
 vector_expansion_->conjugate();
 vector_expansion_->markOptimizableAllTensors();
 exatn::TensorNetworkReconstructor::resetDebugLevel(1,0); //debug
 exatn::TensorNetworkReconstructor reconstructor(opvec_expansion_,vector_expansion_,tolerance_);
 success = exatn::sync(); assert(success);
 double residual_norm, fidelity;
 bool reconstructed = reconstructor.reconstruct(process_group,&residual_norm,&fidelity,true,true);
 success = exatn::sync(); assert(success);
 if(reconstructed){
  if(TensorNetworkLinearSolver::debug > 0)
   std::cout << "Linear solve reconstruction succeeded: Residual norm = " << residual_norm
             << "; Fidelity = " << fidelity << std::endl;

 }else{
  std::cout << "#ERROR(exatn::TensorNetworkLinearSolver): Reconstruction failed!" << std::endl;
 }

 //Rescale the solution:
 vector_expansion_->rescale(std::complex<double>{original_norm,0.0});

 return reconstructed;
}


void TensorNetworkLinearSolver::enableParallelization(bool parallel)
{
 parallel_ = parallel;
 return;
}


void TensorNetworkLinearSolver::resetDebugLevel(unsigned int level, int focus_process)
{
 TensorNetworkLinearSolver::debug = level;
 TensorNetworkLinearSolver::focus = focus_process;
 return;
}

} //namespace exatn
