/** ExaTN:: Reconstructs an approximate tensor network expansion for a given tensor network expansion
REVISION: 2021/03/02

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "reconstructor.hpp"

//#include <talshxx.hpp> //debug

#include <unordered_set>
#include <string>

namespace exatn{

unsigned int TensorNetworkReconstructor::debug{0};


TensorNetworkReconstructor::TensorNetworkReconstructor(std::shared_ptr<TensorExpansion> expansion,
                                                       std::shared_ptr<TensorExpansion> approximant,
                                                       double tolerance):
 expansion_(expansion), approximant_(approximant),
 max_iterations_(DEFAULT_MAX_ITERATIONS), epsilon_(DEFAULT_LEARN_RATE), tolerance_(tolerance),
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


bool TensorNetworkReconstructor::reconstruct(double * residual_norm,
                                             double * fidelity,
                                             bool nesterov)
{
 return reconstruct(exatn::getDefaultProcessGroup(), residual_norm, fidelity, nesterov);
}


bool TensorNetworkReconstructor::reconstruct(const ProcessGroup & process_group,
                                             double * residual_norm,
                                             double * fidelity,
                                             bool nesterov)
{
 unsigned int local_rank; //local process rank within the process group
 if(!process_group.rankIsIn(exatn::getProcessRank(),&local_rank)) return true; //process is not in the group: Do nothing

 assert(residual_norm != nullptr);
 assert(fidelity != nullptr);

 input_norm_ = 0.0;
 output_norm_ = 0.0;
 residual_norm_ = 0.0;
 fidelity_ = 0.0;

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
 bool success = lagrangian.appendExpansion(normalization,{1.0,0.0}); assert(success);
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
 if(!converged){
  //Compute the 2-norm of the input tensor network expansion:
  auto scalar_norm = makeSharedTensor("_scalar_norm");
  bool done = createTensorSync(scalar_norm,environments_[0].tensor->getElementType()); assert(done);
  done = initTensorSync("_scalar_norm",0.0); assert(done);
  done = evaluateSync(process_group,input_norm,scalar_norm); assert(done);
  done = computeNorm1Sync("_scalar_norm",input_norm_); assert(done);
  input_norm_ = std::sqrt(input_norm_);
  if(TensorNetworkReconstructor::debug > 0){
   std::cout << "#DEBUG(exatn::TensorNetworkReconstructor): 2-norm of the input tensor network expansion = "
             << std::scientific << input_norm_ << std::endl;
  }
  //Iterate:
  unsigned int iteration = 0;
  while((!converged) && (iteration < max_iterations_)){
   if(TensorNetworkReconstructor::debug > 0)
    std::cout << "#DEBUG(exatn::TensorNetworkReconstructor): Iteration " << iteration << std::endl;
   double max_grad_norm = 0.0;
   for(auto & environment: environments_){
    //Nesterov extrapolation:
    if(nesterov){
     double extra_coef = static_cast<double>(iteration) / (static_cast<double>(iteration) + 3.0);
     done = scaleTensorSync(environment.tensor->getName(),(1.0+extra_coef)); assert(done);
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
    done = evaluateSync(process_group,environment.gradient_expansion,environment.gradient); assert(done);
    //Compute the norm of the gradient tensor:
    double grad_norm = 0.0;
    done = computeNorm2Sync(environment.gradient->getName(),grad_norm); assert(done);
    //Compute the tensor norm:
    double tens_norm = 0.0;
    done = computeNorm2Sync(environment.tensor->getName(),tens_norm); assert(done);
    double relative_grad_norm = grad_norm / tens_norm;
    if(relative_grad_norm > max_grad_norm) max_grad_norm = relative_grad_norm;
    if(TensorNetworkReconstructor::debug > 1) std::cout << " Gradient norm w.r.t. " << environment.tensor->getName()
                                                        << " = " << grad_norm << ": Tensor norm =  " << tens_norm
                                                        << ": Ratio = " << relative_grad_norm << std::endl;
    //Update the optimizable tensor using the computed gradient:
    if(relative_grad_norm > tolerance_){
     //Compute the optimal step size:
     done = initTensorSync("_scalar_norm",0.0); assert(done);
     done = evaluateSync(process_group,environment.hessian_expansion,scalar_norm); assert(done);
     double hess_grad = 0.0;
     done = computeNorm1Sync("_scalar_norm",hess_grad); assert(done);
     if(hess_grad > 0.0){
      epsilon_ = grad_norm * grad_norm / hess_grad;
     }else{
      epsilon_ = DEFAULT_LEARN_RATE;
     }
     if(TensorNetworkReconstructor::debug > 1) std::cout << " Optimal step size = " << epsilon_ << std::endl;
     //Perform update:
     std::string add_pattern;
     done = generate_addition_pattern(environment.tensor->getRank(),add_pattern,false,
                                      environment.tensor->getName(),environment.gradient->getName()); assert(done);
     done = addTensorsSync(add_pattern,-epsilon_); assert(done);
    }
    //Destroy the gradient tensor:
    done = destroyTensorSync(environment.gradient->getName()); assert(done);
   }
   //Compute the residual norm and check convergence:
   done = initTensorSync("_scalar_norm",0.0); assert(done);
   done = evaluateSync(process_group,residual,scalar_norm); assert(done);
   residual_norm_ = 0.0;
   done = computeNorm1Sync("_scalar_norm",residual_norm_); assert(done);
   residual_norm_ = std::sqrt(residual_norm_);
   if(TensorNetworkReconstructor::debug > 0) std::cout << " Residual norm = " << residual_norm_ << std::endl;
   converged = (max_grad_norm <= tolerance_);
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
  done = evaluateSync(process_group,normalization,scalar_norm); assert(done);
  output_norm_ = 0.0;
  done = computeNorm1Sync("_scalar_norm",output_norm_); assert(done);
  output_norm_ = std::sqrt(output_norm_);
  if(TensorNetworkReconstructor::debug > 0)
   std::cout << "#DEBUG(exatn::TensorNetworkReconstructor): 2-norm of the output tensor network expansion = "
             << output_norm_ << std::endl;
  //Compute approximation fidelity:
  double overlap_abs = 0.0;
  done = initTensorSync("_scalar_norm",0.0); assert(done);
  done = evaluateSync(process_group,overlap_conj,scalar_norm); assert(done);
  done = computeNorm1Sync("_scalar_norm",overlap_abs); assert(done);
  if(TensorNetworkReconstructor::debug > 0)
   std::cout << "#DEBUG(exatn::TensorNetworkReconstructor): Conjugated absolute overlap = " << overlap_abs << std::endl;
  done = initTensorSync("_scalar_norm",0.0); assert(done);
  done = evaluateSync(process_group,overlap,scalar_norm); assert(done);
  done = computeNorm1Sync("_scalar_norm",overlap_abs); assert(done);
  if(TensorNetworkReconstructor::debug > 0)
   std::cout << "#DEBUG(exatn::TensorNetworkReconstructor): Direct absolute overlap = " << overlap_abs << std::endl;
  fidelity_ = std::pow(overlap_abs / (input_norm_ * output_norm_), 2.0);
  done = destroyTensorSync("_scalar_norm"); assert(done);
  //Balance the approximant:
  done = balanceNorm2Sync(process_group,*approximant_,1.0,true); assert(done);
  done = normalizeNorm2Sync(process_group,*approximant_,output_norm_); assert(done);
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


void TensorNetworkReconstructor::resetDebugLevel(unsigned int level)
{
 TensorNetworkReconstructor::debug = level;
 return;
}

} //namespace exatn
