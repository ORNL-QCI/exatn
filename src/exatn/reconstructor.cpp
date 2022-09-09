/** ExaTN:: Reconstructs an approximate tensor network expansion for a given tensor network expansion
REVISION: 2022/08/30

Copyright (C) 2018-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle) **/

#include "reconstructor.hpp"

//#include <talshxx.hpp> //debug

#include <unordered_set>
#include <string>
#include <iostream>
#include <cmath>

namespace exatn{

unsigned int TensorNetworkReconstructor::debug{0};
int TensorNetworkReconstructor::focus{-1};


TensorNetworkReconstructor::TensorNetworkReconstructor(std::shared_ptr<TensorExpansion> expansion,
                                                       std::shared_ptr<TensorExpansion> approximant,
                                                       double tolerance):
 expansion_(expansion), approximant_(approximant),
 max_iterations_(DEFAULT_MAX_ITERATIONS), epsilon_(DEFAULT_LEARN_RATE), tolerance_(tolerance),
#ifdef MPI_ENABLED
 parallel_(true),
#else
 parallel_(false),
#endif
 input_norm_(0.0), output_norm_(0.0), residual_norm_(0.0), fidelity_(0.0)
{
 if(!expansion_->isKet()){
  std::cout << "#ERROR(exatn:TensorNetworkReconstructor): The reconstructed tensor network expansion must be a ket!"
            << std::endl << std::flush;
  assert(false);
 }
 if(!approximant_->isBra()){
  std::cout << "#ERROR(exatn:TensorNetworkReconstructor): The reconstructing tensor network expansion must be a bra!"
            << std::endl << std::flush;
  assert(false);
 }
 if(expansion_->getRank() != approximant_->getRank()){
  std::cout << "#ERROR(exatn:TensorNetworkReconstructor): Rank mismatch in the provided tensor network expansions!"
            << std::endl << std::flush;
  assert(false);
 }
}


void TensorNetworkReconstructor::resetTolerance(double tolerance)
{
 tolerance_ = tolerance;
 return;
}


void TensorNetworkReconstructor::resetLearningRate(double learn_rate)
{
 epsilon_ = learn_rate;
 return;
}


void TensorNetworkReconstructor::resetMaxIterations(unsigned int max_iterations)
{
 max_iterations_ = max_iterations;
 return;
}


std::shared_ptr<TensorExpansion> TensorNetworkReconstructor::getSolution(double * residual_norm,
                                                                         double * fidelity) const
{
 if(fidelity_ == 0.0) return std::shared_ptr<TensorExpansion>(nullptr);
 *residual_norm = residual_norm_;
 *fidelity = fidelity_;
 return approximant_;
}


void TensorNetworkReconstructor::reinitializeApproximant(const ProcessGroup & process_group)
{
 assert(approximant_);
 bool success = true;
 std::unordered_set<std::string> tensor_names;
 for(auto component = approximant_->begin(); component != approximant_->end(); ++component){
  auto & network = *(component->network);
  for(auto tens = network.begin(); tens != network.end(); ++tens){
   if(tens->first != 0){
    const auto & tens_name = tens->second.getName();
    auto res = tensor_names.emplace(tens_name);
    if(res.second){
     success = exatn::initTensorRnd(tens_name); assert(success);
    }
   }
  }
 }
 success = exatn::sync(process_group); assert(success);
 return;
}


bool TensorNetworkReconstructor::reconstruct(double * residual_norm,
                                             double * fidelity,
                                             bool rnd_init,
                                             bool nesterov,
                                             double acceptable_fidelity)
{
 return reconstruct(exatn::getDefaultProcessGroup(), residual_norm, fidelity, rnd_init, nesterov, acceptable_fidelity);
}


bool TensorNetworkReconstructor::reconstruct(const ProcessGroup & process_group,
                                             double * residual_norm,
                                             double * fidelity,
                                             bool rnd_init,
                                             bool nesterov,
                                             double acceptable_fidelity)
{
 return reconstruct_sd(process_group, residual_norm, fidelity, rnd_init, nesterov, acceptable_fidelity);
}


bool TensorNetworkReconstructor::reconstruct_sd(const ProcessGroup & process_group,
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

 assert(residual_norm != nullptr);
 assert(fidelity != nullptr);

 if(TensorNetworkReconstructor::focus >= 0){
  if(getProcessRank() != TensorNetworkReconstructor::focus) TensorNetworkReconstructor::debug = 0;
 }

 input_norm_ = 0.0;
 output_norm_ = 0.0;
 residual_norm_ = 0.0;
 fidelity_ = 0.0;

 //Initialize the approximant to a random value (if needed):
 if(rnd_init) reinitializeApproximant(process_group);

 //Balance-normalize the approximant (only optimizable tensors):
 bool success = balanceNormalizeNorm2Sync(process_group,*approximant_,1.0,1.0,true); assert(success);

 if(TensorNetworkReconstructor::debug > 0){
  std::cout << "#DEBUG(exatn::TensorNetworkReconstructor): Target tensor expansion:" << std::endl;
  expansion_->printIt();
  std::cout << "#DEBUG(exatn::TensorNetworkReconstructor): Approximant tensor expansion:" << std::endl;
  approximant_->printIt();
 }

 //Activate caching of optimal tensor contraction sequences:
 bool con_seq_caching = queryContrSeqCaching();
 if(!con_seq_caching) activateContrSeqCaching();

 //Construct the Lagrangian optimization functional (scalar):
 // <approximant|approximant> - <approximant|expansion>
 TensorExpansion approximant_ket(*approximant_,false); // <approximant|
 approximant_ket.conjugate(); // |approximant>
 TensorExpansion overlap(*approximant_,*expansion_); // <approximant|expansion>
 overlap.rename("Overlap");
 TensorExpansion normalization(*approximant_,approximant_ket); // <approximant|approximant>
 normalization.rename("Normalization");
 TensorExpansion lagrangian;
 success = lagrangian.appendExpansion(normalization,{1.0,0.0}); assert(success);
 success = lagrangian.appendExpansion(overlap,{-1.0,0.0}); assert(success);
 lagrangian.rename("Lagrangian");
 if(TensorNetworkReconstructor::debug > 1){
  std::cout << "#DEBUG(exatn::TensorNetworkReconstructor): Lagrangian:" << std::endl;
  lagrangian.printIt();
 }

 //Construct the residual functional (real scalar cost function):
 // <expansion|expansion> + <approximant|approximant> - <approximant|expansion> - <expansion|approximant>
 TensorExpansion expansion_bra(*expansion_,false); // |expansion>
 expansion_bra.conjugate(); // <expansion|
 TensorExpansion input_norm(expansion_bra,*expansion_); // <expansion|expansion>
 input_norm.rename("InputNorm");
 TensorExpansion overlap_conj(expansion_bra,approximant_ket); // <expansion|approximant>
 overlap_conj.rename("OverlapConj");
 TensorExpansion residual;
 success = residual.appendExpansion(input_norm,{1.0,0.0}); assert(success);
 success = residual.appendExpansion(normalization,{1.0,0.0}); assert(success);
 success = residual.appendExpansion(overlap,{-1.0,0.0}); assert(success);
 success = residual.appendExpansion(overlap_conj,{-1.0,0.0}); assert(success);
 residual.rename("Residual");
 if(TensorNetworkReconstructor::debug > 1){
  std::cout << "#DEBUG(exatn::TensorNetworkReconstructor): Residual:" << std::endl;
  residual.printIt();
 }

 //Prepare derivative environments for all optimizable tensors in the approximant:
 std::unordered_set<std::string> tensor_names;
 // Loop over the tensor networks constituting the approximant tensor expansion:
 for(auto network = approximant_->cbegin(); network != approximant_->cend(); ++network){
  // Loop over the optimizable tensors inside the current tensor network:
  for(auto tensor_conn = network->network->begin(); tensor_conn != network->network->end(); ++tensor_conn){
   const auto & tensor = tensor_conn->second;
   if(tensor.isOptimizable()){ //gradient w.r.t. an optimizable tensor inside the approximant tensor expansion
    auto res = tensor_names.emplace(tensor.getName());
    if(res.second){ //prepare derivative environment only once for each unique tensor name
     auto gradient_tensor = std::make_shared<Tensor>("_g"+tensor.getName(),tensor.getShape(),tensor.getSignature());
     environments_.emplace_back(Environment{tensor.getTensor(),                             //optimizable tensor
                                            std::make_shared<Tensor>("_a"+tensor.getName(), //auxiliary tensor
                                                                     tensor.getShape(),
                                                                     tensor.getSignature()),
                                            gradient_tensor,                                //gradient tensor
                                            std::make_shared<Tensor>("_h"+tensor.getName(), //auxiliary gradient tensor
                                                                     tensor.getShape(),
                                                                     tensor.getSignature()),
                                            TensorExpansion(lagrangian,tensor.getName(),true), //derivative tensor network expansion w.r.t. conjugated (bra) tensor
                                            TensorExpansion(normalization,tensor.getTensor(),gradient_tensor)});
     if(nesterov){
      bool done = createTensorSync(environments_.back().tensor_aux,environments_[0].tensor->getElementType()); assert(done);
      done = initTensorSync(environments_.back().tensor_aux->getName(),0.0); assert(done);
     }
    }
   }
  }
 }
 if(TensorNetworkReconstructor::debug > 1){
  std::cout << "#DEBUG(exatn::TensorNetworkReconstructor): Derivatives:" << std::endl;
  for(const auto & environment: environments_){
   std::cout << "#DEBUG: Derivative tensor network expansion w.r.t. " << environment.tensor->getName() << std::endl;
   environment.gradient_expansion.printIt();
  }
 }

 //Tensor optimization procedure:
 bool converged = environments_.empty();
 while(!converged){
  double overlap_abs = 0.0;
  double overlap_prev = 0.0;
  double overlap_diff = 0.0;
  if(TensorNetworkReconstructor::debug > 0) expansion_->printCoefficients();
  //Compute the 2-norm of the input tensor network expansion:
  auto scalar_norm = makeSharedTensor("_scalar_norm");
  bool done = createTensorSync(scalar_norm,environments_[0].tensor->getElementType()); assert(done);
  done = initTensorSync("_scalar_norm",0.0); assert(done);
  done = evaluateSync(process_group,input_norm,scalar_norm,num_procs); assert(done);
  input_norm_ = 0.0;
  done = computeNorm1Sync("_scalar_norm",input_norm_); assert(done);
  input_norm_ = std::sqrt(input_norm_);
  //Check the initial guess:
  overlap_abs = 0.0;
  while(overlap_abs <= DEFAULT_MIN_INITIAL_OVERLAP){
   //Compute the approximant norm:
   done = initTensorSync("_scalar_norm",0.0); assert(done);
   done = evaluateSync(process_group,normalization,scalar_norm,num_procs); assert(done);
   output_norm_ = 0.0;
   done = computeNorm1Sync("_scalar_norm",output_norm_); assert(done);
   output_norm_ = std::sqrt(output_norm_);
   //Compute the direct absolute overlap with the approximant:
   done = initTensorSync("_scalar_norm",0.0); assert(done);
   done = evaluateSync(process_group,overlap,scalar_norm,num_procs); assert(done);
   overlap_abs = 0.0;
   done = computeNorm1Sync("_scalar_norm",overlap_abs); assert(done);
   overlap_abs /= (output_norm_ * input_norm_);
   if(TensorNetworkReconstructor::debug > 0){
    std::cout << "#DEBUG(exatn::TensorNetworkReconstructor): Input 2-norm = "
              << std::scientific << input_norm_ << "; Output 2-norm = " << output_norm_
              << "; Absolute overlap = " << overlap_abs << std::endl;
   }
   if(overlap_abs <= DEFAULT_MIN_INITIAL_OVERLAP){
    if(TensorNetworkReconstructor::debug > 0)
     std::cout << "#DEBUG(exatn::TensorNetworkReconstructor): Approximant will be reinitialized due to low overlap\n";
    reinitializeApproximant(process_group);
   }
  }
  //Iterate:
  unsigned int iteration = 0;
  while((!converged) && (iteration < max_iterations_)){
   if(TensorNetworkReconstructor::debug > 0)
    std::cout << "#DEBUG(exatn::TensorNetworkReconstructor)["
              << std::fixed << std::setprecision(6) << exatn::Timer::timeInSecHR(numericalServer->getTimeStampStart())
              << "]: Iteration " << iteration << std::endl;
   double max_grad_norm = 0.0;
   for(auto & environment: environments_){
    //Nesterov extrapolation:
    if(nesterov){
     double extra_coef = static_cast<double>(iteration) / (static_cast<double>(iteration) + 3.0);
     done = scaleTensorSync(environment.tensor->getName(),(1.0 + extra_coef)); assert(done);
     std::string add_pattern;
     done = generate_addition_pattern(environment.tensor->getRank(),add_pattern,false,
                                      environment.tensor->getName(),environment.tensor_aux->getName()); assert(done);
     done = addTensorsSync(add_pattern,-extra_coef); assert(done);
     done = scaleTensorSync(environment.tensor_aux->getName(),extra_coef/(1.0+extra_coef)); assert(done);
     add_pattern.clear();
     done = generate_addition_pattern(environment.tensor->getRank(),add_pattern,false,
                                      environment.tensor_aux->getName(),environment.tensor->getName()); assert(done);
     done = addTensorsSync(add_pattern,1.0/(1.0+extra_coef)); assert(done);
    }
    //Create the gradient tensor:
    done = createTensorSync(environment.gradient,environment.tensor->getElementType()); assert(done);
    //Initialize the gradient tensor to zero:
    done = initTensorSync(environment.gradient->getName(),0.0); assert(done);
    //Evaluate the gradient tensor expansion:
    done = evaluateSync(process_group,environment.gradient_expansion,environment.gradient,num_procs); assert(done);
    //Compute the norm of the gradient tensor:
    double grad_norm = 0.0;
    done = computeNorm2Sync(environment.gradient->getName(),grad_norm); assert(done);
    if(TensorNetworkReconstructor::debug > 1){
     std::cout << environment.tensor->getName()
               << ": Raw gradient norm = " << std::scientific << grad_norm;
    }
    assert(!std::isnan(grad_norm));
    //Compute the tensor norm:
    double tens_norm = 0.0;
    done = computeNorm2Sync(environment.tensor->getName(),tens_norm); assert(done);
    assert(tens_norm > 1e-7);
    double relative_grad_norm = grad_norm / tens_norm;
    //Update the optimizable tensor using the computed gradient:
    //Compute the optimal step size:
    done = initTensorSync("_scalar_norm",0.0); assert(done);
    done = evaluateSync(process_group,environment.hessian_expansion,scalar_norm,num_procs); assert(done);
    double hess_grad = 0.0;
    done = computeNorm1Sync("_scalar_norm",hess_grad); assert(done);
    if(TensorNetworkReconstructor::debug > 1){
     std::cout << "; Raw hessian norm = " << std::scientific << std::sqrt(hess_grad);
    }
    if(hess_grad > 0.0){
     epsilon_ = grad_norm * grad_norm / hess_grad; //Cauchy step size
    }else{
     epsilon_ = DEFAULT_LEARN_RATE;
    }
    max_grad_norm = std::max(max_grad_norm,relative_grad_norm);
    if(TensorNetworkReconstructor::debug > 1){
     std::cout << "; Normalized gradient norm = " << std::scientific << relative_grad_norm
               << ": Tensor norm = " << tens_norm
               << ": Optimal step size = " << epsilon_ << std::endl;
    }
    //Update the optimizable tensor:
    std::string add_pattern;
    done = generate_addition_pattern(environment.tensor->getRank(),add_pattern,false,
                                     environment.tensor->getName(),environment.gradient->getName()); assert(done);
    done = addTensorsSync(add_pattern,-epsilon_); assert(done);
    //Check the norm of the updated tensor:
    double new_tens_norm = 0.0;
    done = computeNorm2Sync(environment.tensor->getName(),new_tens_norm); assert(done);
    if(std::abs(new_tens_norm) < 1e-13){
     if(TensorNetworkReconstructor::debug > 0){
      std::cout << "#EXCEPTION(exatn::reconstructor): Tensor update to zero!\n";
      //printTensor(environment.tensor->getName());
      //printTensor(environment.gradient->getName());
     }
     done = addTensorsSync(add_pattern,epsilon_*0.5); assert(done);
    }
    //Normalize the optimizable tensor to unity:
    done = normalizeNorm2Sync(environment.tensor->getName(),1.0); assert(done);
    //Destroy the gradient tensor:
    done = destroyTensorSync(environment.gradient->getName()); assert(done);
   }
   //Compute the residual norm and check convergence:
   done = initTensorSync("_scalar_norm",0.0); assert(done);
   done = evaluateSync(process_group,residual,scalar_norm,num_procs); assert(done);
   residual_norm_ = 0.0;
   done = computeNorm1Sync("_scalar_norm",residual_norm_); assert(done);
   residual_norm_ = std::sqrt(residual_norm_);
   if(TensorNetworkReconstructor::debug > 0){
    std::cout << " Residual norm = " << std::scientific << residual_norm_
              << "; Max gradient = " << max_grad_norm << std::endl;
    approximant_->printCoefficients();
   }
   //Compute the approximant norm:
   done = initTensorSync("_scalar_norm",0.0); assert(done);
   done = evaluateSync(process_group,normalization,scalar_norm,num_procs); assert(done);
   output_norm_ = 0.0;
   done = computeNorm1Sync("_scalar_norm",output_norm_); assert(done);
   output_norm_ = std::sqrt(output_norm_);
   //Compute the direct overlap:
   done = initTensorSync("_scalar_norm",0.0); assert(done);
   done = evaluateSync(process_group,overlap,scalar_norm,num_procs); assert(done);
   overlap_abs = 0.0;
   done = computeNorm1Sync("_scalar_norm",overlap_abs); assert(done);
   overlap_abs = overlap_abs / (output_norm_ * input_norm_);
   overlap_diff = std::max(overlap_diff,std::abs(overlap_abs-overlap_prev));
   overlap_prev = overlap_abs;
   if(TensorNetworkReconstructor::debug > 0)
    std::cout << "#DEBUG(exatn::TensorNetworkReconstructor): 2-norm of the output tensor network expansion = "
              << std::scientific << output_norm_ << "; Absolute overlap = " << overlap_abs << std::endl;
   bool last_diff_iteration = (iteration % DEFAULT_OVERLAP_ITERATIONS == (DEFAULT_OVERLAP_ITERATIONS-1));
   converged = (overlap_diff/overlap_abs <= tolerance_) && last_diff_iteration;
   //converged = (max_grad_norm <= tolerance_) || ((overlap_diff/overlap_abs <= tolerance_) && last_diff_iteration);
   if(last_diff_iteration) overlap_diff = 0.0;
   ++iteration;
  }
  /*//Inspect the residual norm contributions (debug):
  if(TensorNetworkReconstructor::debug > 1){
   std::cout << "#DEBUG(exatn::TensorNetworkReconstructor): Individual components of residual:";
   for(auto net_iter = residual.cbegin(); net_iter != residual.cend(); ++net_iter){
    auto output_tensor = exatn::getLocalTensor(net_iter->network->getTensor(0)->getName());
    auto view = output_tensor->getSliceView<float>();
    std::cout << " " << view[std::initializer_list<int>{}];
   }
   std::cout << std::endl;
  }*/
  //Compute the approximant norm:
  done = initTensorSync("_scalar_norm",0.0); assert(done);
  done = evaluateSync(process_group,normalization,scalar_norm,num_procs); assert(done);
  output_norm_ = 0.0;
  done = computeNorm1Sync("_scalar_norm",output_norm_); assert(done);
  output_norm_ = std::sqrt(output_norm_);
  if(TensorNetworkReconstructor::debug > 0)
   std::cout << "#DEBUG(exatn::TensorNetworkReconstructor): 2-norm of the output tensor network expansion = "
             << std::scientific << output_norm_ << std::endl;
  //Compute final approximation fidelity and overlap:
  done = initTensorSync("_scalar_norm",0.0); assert(done);
  done = evaluateSync(process_group,overlap_conj,scalar_norm,num_procs); assert(done);
  overlap_abs = 0.0;
  done = computeNorm1Sync("_scalar_norm",overlap_abs); assert(done);
  if(TensorNetworkReconstructor::debug > 0)
   std::cout << "#DEBUG(exatn::TensorNetworkReconstructor): Conjugated overlap = "
             << std::scientific << overlap_abs << std::endl;
  done = initTensorSync("_scalar_norm",0.0); assert(done);
  done = evaluateSync(process_group,overlap,scalar_norm,num_procs); assert(done);
  overlap_abs = 0.0;
  done = computeNorm1Sync("_scalar_norm",overlap_abs); assert(done);
  fidelity_ = std::pow(overlap_abs / (output_norm_ * input_norm_), 2.0);
  if(TensorNetworkReconstructor::debug > 0){
   std::cout << "#DEBUG(exatn::TensorNetworkReconstructor): Direct overlap = "
             << std::scientific << overlap_abs << std::endl;
   std::cout << "#DEBUG(exatn::TensorNetworkReconstructor): Absolute overlap = "
             << std::scientific << (overlap_abs / (output_norm_ * input_norm_)) << std::endl;
   std::cout << "#DEBUG(exatn::TensorNetworkReconstructor): Fidelity = "
             << std::scientific  << fidelity_ << std::endl;
  }
  done = destroyTensorSync("_scalar_norm"); assert(done);
  //Check the necessity to restart iterations:
  if(iteration >= max_iterations_) break;
  if(converged && fidelity_ < acceptable_fidelity && tolerance_ > 1e-6){
   if(TensorNetworkReconstructor::debug > 0){
    std::cout << "#DEBUG(exatn::TensorNetworkReconstructor): Insufficient fidelity, iterations will be restarted\n";
   }
   tolerance_ *= 0.1;
   converged = false;
  }
 }

 //Destroy auxiliary tensors:
 if(nesterov){
  for(auto & environment: environments_){
   bool done = destroyTensorSync(environment.tensor_aux->getName()); assert(done);
  }
 }

 //Deactivate caching of optimal tensor contraction sequences:
 if(!con_seq_caching) deactivateContrSeqCaching();

 *residual_norm = residual_norm_;
 *fidelity = fidelity_;
 return converged;
}


void TensorNetworkReconstructor::enableParallelization(bool parallel)
{
 parallel_ = parallel;
 return;
}


void TensorNetworkReconstructor::resetDebugLevel(unsigned int level, int focus_process)
{
 TensorNetworkReconstructor::debug = level;
 TensorNetworkReconstructor::focus = focus_process;
 return;
}

} //namespace exatn
