/** ExaTN:: Reconstructs an approximate tensor network operator for a given tensor network operator
REVISION: 2021/10/22

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "remapper.hpp"
#include "reconstructor.hpp"

#include <string>
#include <iostream>

namespace exatn{

unsigned int TensorOperatorRemapper::debug{0};
int TensorOperatorRemapper::focus{-1};


TensorOperatorRemapper::TensorOperatorRemapper(std::shared_ptr<Tensor> ket_space,
                                               std::shared_ptr<Tensor> bra_space,
                                               std::shared_ptr<TensorOperator> target,
                                               std::shared_ptr<TensorOperator> approximant,
                                               double tolerance):
 ket_space_(ket_space), bra_space_(bra_space), target_(target), approximant_(approximant),
 max_iterations_(DEFAULT_MAX_ITERATIONS), tolerance_(tolerance),
#ifdef MPI_ENABLED
 parallel_(true),
#else
 parallel_(false),
#endif
 residual_norm_(0.0), fidelity_(0.0)
{
}


TensorOperatorRemapper::TensorOperatorRemapper(std::shared_ptr<Tensor> ket_space,
                                               std::shared_ptr<TensorOperator> target,
                                               std::shared_ptr<TensorOperator> approximant,
                                               double tolerance):
 ket_space_(ket_space), bra_space_(ket_space), target_(target), approximant_(approximant),
 max_iterations_(DEFAULT_MAX_ITERATIONS), tolerance_(tolerance),
#ifdef MPI_ENABLED
 parallel_(true),
#else
 parallel_(false),
#endif
 residual_norm_(0.0), fidelity_(0.0)
{
}


void TensorOperatorRemapper::resetTolerance(double tolerance)
{
 tolerance_ = tolerance;
 return;
}


void TensorOperatorRemapper::resetMaxIterations(unsigned int max_iterations)
{
 max_iterations_ = max_iterations;
 return;
}


std::shared_ptr<TensorOperator> TensorOperatorRemapper::getSolution(double * residual_norm,
                                                                    double * fidelity) const
{
 if(fidelity_ == 0.0) return std::shared_ptr<TensorOperator>(nullptr);
 *residual_norm = residual_norm_;
 *fidelity = fidelity_;
 return approximant_;
}


bool TensorOperatorRemapper::reconstruct(double * residual_norm,
                                         double * fidelity,
                                         bool rnd_init,
                                         bool nesterov,
                                         double acceptable_fidelity)
{
 return reconstruct(exatn::getDefaultProcessGroup(), residual_norm, fidelity, rnd_init, nesterov, acceptable_fidelity);
}


bool TensorOperatorRemapper::reconstruct(const ProcessGroup & process_group,
                                         double * residual_norm,
                                         double * fidelity,
                                         bool rnd_init,
                                         bool nesterov,
                                         double acceptable_fidelity)
{
 unsigned int local_rank; //local process rank within the process group
 if(!process_group.rankIsIn(exatn::getProcessRank(),&local_rank)) return true; //process is not in the group: Do nothing
 unsigned int num_procs = 1;
 if(parallel_) num_procs = process_group.getSize();

 if(TensorOperatorRemapper::focus >= 0){
  if(getProcessRank() != TensorOperatorRemapper::focus) TensorOperatorRemapper::debug = 0;
 }

 //Remap tensor network operators as tensor network expansions in the given space:
 auto target_expansion = makeSharedTensorExpansion(*target_,*ket_space_,*bra_space_);
 auto approx_expansion = makeSharedTensorExpansion(*approximant_,*ket_space_,*bra_space_);

 //Normalize the target tensor network expansion to unity:
 double original_norm = 0.0;
 bool success = normalizeNorm2Sync(process_group,*target_expansion,1.0,&original_norm); assert(success);
 if(TensorOperatorRemapper::debug > 0)
  std::cout << "#DEBUG(exatn::TensorOperatorRemapper): Original target norm = " << original_norm << std::endl;

 //Reconstruct the target via approximant:
 approx_expansion->conjugate();
 approx_expansion->markOptimizableAllTensors();
 exatn::TensorNetworkReconstructor::resetDebugLevel(TensorOperatorRemapper::debug,
                                                    TensorOperatorRemapper::focus);
 exatn::TensorNetworkReconstructor reconstructor(target_expansion,approx_expansion,tolerance_);
 success = exatn::sync(); assert(success);
 bool reconstructed = reconstructor.reconstruct(process_group,&residual_norm_,&fidelity_,true,true);
 success = exatn::sync(); assert(success);
 if(reconstructed){
  if(TensorOperatorRemapper::debug > 0)
   std::cout << "Tensor operator reconstruction succeeded: Residual norm = " << residual_norm_
             << "; Fidelity = " << fidelity_ << std::endl;
 }else{
  std::cout << "#ERROR(exatn::TensorOperatorRemapper): Reconstruction failed!" << std::endl;
 }

 //Rescale the solution:
 approx_expansion->conjugate();
 approx_expansion->rescale(std::complex<double>{original_norm,0.0});
 const auto num_components = approximant_->getNumComponents();
 assert(approx_expansion->getNumComponents() == num_components);
 for(std::size_t i = 0; i < num_components; ++i){
  (*approximant_)[i].coefficient = (*approx_expansion)[i].coefficient;
 }

 return reconstructed;
}


void TensorOperatorRemapper::enableParallelization(bool parallel)
{
 parallel_ = parallel;
 return;
}


void TensorOperatorRemapper::resetDebugLevel(unsigned int level, int focus_process)
{
 TensorOperatorRemapper::debug = level;
 TensorOperatorRemapper::focus = focus_process;
 return;
}

} //namespace exatn
