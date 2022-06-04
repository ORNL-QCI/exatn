/** ExaTN:: Variational optimizer of a closed symmetric tensor network expansion functional
REVISION: 2022/06/03

Copyright (C) 2018-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle)
Copyright (C) 2022-2022 NVIDIA Corporation **/

#include "optimizer.hpp"

#include <talshxx.hpp>

#include <unordered_set>
#include <string>
#include <iostream>

//LAPACK zggev:
extern "C" {
void zggev_(
 char const * jobvl, char const * jobvr,
 int const * n,
 void * A, int const * lda,
 void * B, int const * ldb,
 void * alpha,
 void * beta,
 void * VL, int const * ldvl,
 void * VR, int const * ldvr,
 void * work, int const * lwork,
 void * rwork,
 int * info);
}

namespace exatn{

unsigned int TensorNetworkOptimizer::debug{0};
int TensorNetworkOptimizer::focus{-1};


TensorNetworkOptimizer::TensorNetworkOptimizer(std::shared_ptr<TensorOperator> tensor_operator,
                                               std::shared_ptr<TensorExpansion> vector_expansion,
                                               double tolerance):
 tensor_operator_(tensor_operator), vector_expansion_(vector_expansion),
 max_iterations_(DEFAULT_MAX_ITERATIONS), micro_iterations_(DEFAULT_MICRO_ITERATIONS),
 epsilon_(DEFAULT_LEARN_RATE), tolerance_(tolerance),
#ifdef MPI_ENABLED
 parallel_(true),
#else
 parallel_(false),
#endif
 average_expect_val_({0.0,0.0})
{
 if(!vector_expansion_->isKet()){
  std::cout << "#ERROR(exatn:TensorNetworkOptimizer): The tensor network vector expansion must be a ket!"
            << std::endl << std::flush;
  assert(false);
 }
}


void TensorNetworkOptimizer::resetTolerance(double tolerance)
{
 tolerance_ = tolerance;
 return;
}


void TensorNetworkOptimizer::resetLearningRate(double learn_rate)
{
 epsilon_ = learn_rate;
 return;
}


void TensorNetworkOptimizer::resetMaxIterations(unsigned int max_iterations)
{
 max_iterations_ = max_iterations;
 return;
}


void TensorNetworkOptimizer::resetMicroIterations(unsigned int micro_iterations)
{
 micro_iterations_ = micro_iterations;
 return;
}


std::shared_ptr<TensorExpansion> TensorNetworkOptimizer::getSolution(std::complex<double> * expect_val) const
{
 if(expect_val != nullptr) *expect_val = average_expect_val_;
 return vector_expansion_;
}


std::shared_ptr<TensorExpansion> TensorNetworkOptimizer::getSolution(unsigned int root_id,
                                                                     std::complex<double> * expect_val) const
{
 assert(root_id < eigenvalues_.size());
 if(expect_val != nullptr) *expect_val = eigenvalues_[root_id];
 return eigenvectors_[root_id];
}


std::complex<double> TensorNetworkOptimizer::getExpectationValue() const
{
 return average_expect_val_;
}


std::complex<double> TensorNetworkOptimizer::getExpectationValue(unsigned int root_id) const
{
 assert(root_id < eigenvalues_.size());
 return eigenvalues_[root_id];
}


bool TensorNetworkOptimizer::optimize(bool multistate)
{
 return optimize(exatn::getDefaultProcessGroup(),multistate);
}


bool TensorNetworkOptimizer::optimize(const ProcessGroup & process_group, bool multistate)
{
 if(multistate) return optimize_tr(process_group);
 return optimize_sd(process_group);
}


bool TensorNetworkOptimizer::optimizeSequential(unsigned int num_roots)
{
 return optimizeSequential(exatn::getDefaultProcessGroup(),num_roots);
}


bool TensorNetworkOptimizer::optimizeSequential(const ProcessGroup & process_group,
                                                unsigned int num_roots)
{
 bool success = true;
 auto original_operator = tensor_operator_;
 for(unsigned int root_id = 0; root_id < num_roots; ++root_id){
  success = initTensorsRndSync(*vector_expansion_); assert(success);
  bool synced = sync(process_group); assert(synced);
  success = optimize(process_group);
  synced = sync(process_group); assert(synced);
  if(!success) break;
  const auto expect_val = getExpectationValue();
  eigenvalues_.emplace_back(expect_val);
  auto solution_vector = duplicateSync(process_group,*vector_expansion_);
  assert(solution_vector);
  success = normalizeNorm2Sync(*solution_vector,1.0); assert(success);
  eigenvectors_.emplace_back(solution_vector);
  for(auto ket_net = solution_vector->begin(); ket_net != solution_vector->end(); ++ket_net){
   ket_net->network->markOptimizableNoTensors();
  }
  const auto num_legs = solution_vector->getRank();
  std::vector<std::pair<unsigned int, unsigned int>> ket_pairing(num_legs);
  for(unsigned int i = 0; i < num_legs; ++i) ket_pairing[i] = std::make_pair(i,i);
  std::vector<std::pair<unsigned int, unsigned int>> bra_pairing(num_legs);
  for(unsigned int i = 0; i < num_legs; ++i) bra_pairing[i] = std::make_pair(i,i);
  auto projector = makeSharedTensorOperator("EigenProjector" + std::to_string(root_id));
  for(auto ket_net = solution_vector->cbegin(); ket_net != solution_vector->cend(); ++ket_net){
   for(auto bra_net = solution_vector->cbegin(); bra_net != solution_vector->cend(); ++bra_net){
    success = projector->appendComponent(ket_net->network,bra_net->network,ket_pairing,bra_pairing,
                                         (-expect_val) * std::conj(ket_net->coefficient) * (bra_net->coefficient));
    assert(success);
   }
  }
  auto proj_hamiltonian = combineTensorOperators(*tensor_operator_,*projector);
  assert(proj_hamiltonian);
  tensor_operator_ = proj_hamiltonian;
 }
 tensor_operator_ = original_operator;
 return success;
}


bool TensorNetworkOptimizer::optimize_sd(const ProcessGroup & process_group)
{
 constexpr bool NORMALIZE_WITH_METRICS = true;  //whether to normalize tensor network factors with metrics or not
 constexpr double MIN_ACCEPTABLE_DENOM = 1e-13; //minimally acceptable denominator in optimal step size determination
 constexpr bool COLLAPSE_ISOMETRIES = true;     //enables collapsing isometries in all tensor networks

 unsigned int local_rank; //local process rank within the process group
 if(!process_group.rankIsIn(exatn::getProcessRank(),&local_rank)) return true; //process is not in the group: Do nothing
 unsigned int num_procs = 1;
 if(parallel_) num_procs = process_group.getSize();

 if(TensorNetworkOptimizer::focus >= 0){
  if(getProcessRank() != TensorNetworkOptimizer::focus) TensorNetworkOptimizer::debug = 0;
 }

 //Pre-optimize the initial tensor network vector expansion guess:
 if(PREOPTIMIZE_INITIAL_GUESS) computeInitialGuess(process_group);

 //Balance-normalize the tensor network vector expansion:
 //bool success = balanceNormalizeNorm2Sync(*vector_expansion_,1.0,1.0,true); assert(success);
 bool success = balanceNorm2Sync(*vector_expansion_,1.0,true); assert(success);

 if(TensorNetworkOptimizer::debug > 0){
  std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Tensor network operator:" << std::endl;
  tensor_operator_->printIt();
  std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Tensor network vector:" << std::endl;
  vector_expansion_->printIt();
 }

 //Activate caching of optimal tensor contraction sequences:
 bool con_seq_caching = queryContrSeqCaching();
 if(!con_seq_caching) activateContrSeqCaching();

 //Construct the operator expectation expansion:
 // <vector|operator|vector>
 TensorExpansion bra_vector_expansion(*vector_expansion_);
 bra_vector_expansion.conjugate();
 bra_vector_expansion.rename(vector_expansion_->getName()+"Bra");
 TensorExpansion operator_expectation(bra_vector_expansion,*vector_expansion_,*tensor_operator_);
 operator_expectation.rename("OperatorExpectation");
 for(auto net = operator_expectation.begin(); net != operator_expectation.end(); ++net){
  net->network->rename("OperExpect" + std::to_string(std::distance(operator_expectation.begin(),net)));
 }
 if(TensorNetworkOptimizer::debug > 1){
  std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Operator expectation expansion:" << std::endl;
  operator_expectation.printIt();
 }

 //Construct the metrics expectation expansion:
 // <vector|vector>
 TensorExpansion metrics_expectation(bra_vector_expansion,*vector_expansion_);
 metrics_expectation.rename("MetricsExpectation");
 for(auto net = metrics_expectation.begin(); net != metrics_expectation.end(); ++net){
  net->network->rename("MetrExpect" + std::to_string(std::distance(metrics_expectation.begin(),net)));
 }
 if(TensorNetworkOptimizer::debug > 1){
  std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Metrics expectation expansion:" << std::endl;
  metrics_expectation.printIt();
 }

 //Construct the residual expectation expansion:
 // <vector|operator|vector> - <vector|vector>
 TensorExpansion residual_expectation;
 success = residual_expectation.appendExpansion(operator_expectation,{1.0,0.0}); assert(success);
 success = residual_expectation.appendExpansion(metrics_expectation,{-1.0,0.0}); assert(success);
 residual_expectation.rename("ResidualExpectation");
 if(TensorNetworkOptimizer::debug > 1){
  std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Residual expectation expansion:" << std::endl;
  residual_expectation.printIt();
 }

 //Prepare derivative environments for all optimizable tensors in the vector expansion:
 environments_.clear();
 std::unordered_set<std::string> tensor_names;
 // Loop over the tensor networks constituting the tensor network vector expansion:
 for(auto network = vector_expansion_->cbegin(); network != vector_expansion_->cend(); ++network){
  // Loop over the optimizable tensors inside the current tensor network:
  for(auto tensor_conn = network->network->begin(); tensor_conn != network->network->end(); ++tensor_conn){
   const auto & tensor = tensor_conn->second;
   if(tensor.isOptimizable()){ //gradient w.r.t. an optimizable tensor inside the tensor network vector expansion
    auto res = tensor_names.emplace(tensor.getName());
    if(res.second){ //prepare derivative environment only once for each unique tensor name
     auto gradient_tensor = std::make_shared<Tensor>("_g"+tensor.getName(),tensor.getShape(),tensor.getSignature());
     std::vector<TensorLeg> iso_self_pattern;
     auto overlap_tensor = std::make_shared<Tensor>("_s"+tensor.getName(),*(tensor.getTensor()),iso_self_pattern,0);
     environments_.emplace_back(Environment{tensor.getTensor(),                              //optimizable tensor
                                            gradient_tensor,                                 //gradient tensor
                                            std::make_shared<Tensor>("_h"+tensor.getName(),  //auxiliary gradient tensor
                                                                     tensor.getShape(),
                                                                     tensor.getSignature()),
                                            overlap_tensor,                                  //gradient-tensor overlap
                                            TensorExpansion(residual_expectation,tensor.getName(),true), // |operator|tensor> - |metrics|tensor>
                                            TensorExpansion(operator_expectation,tensor.getName(),true), // |operator|tensor>
                                            TensorExpansion(metrics_expectation,tensor.getName(),true),  // |metrics|tensor>
                                            TensorExpansion(residual_expectation,tensor.getTensor(),gradient_tensor), // <gradient|operator|gradient> - <gradient|metrics|gradient>
                                            {1.0,0.0}});
     if(COLLAPSE_ISOMETRIES){
      auto collapsed = environments_.back().gradient_expansion.collapseIsometries();
      collapsed = environments_.back().operator_gradient.collapseIsometries();
      collapsed = environments_.back().metrics_gradient.collapseIsometries();
      collapsed = environments_.back().hessian_expansion.collapseIsometries();
     }
    }
   }
  }
 }
 //Collapse isometries in the original TN functionals:
 if(COLLAPSE_ISOMETRIES){
  auto collapsed = operator_expectation.collapseIsometries();
  collapsed = metrics_expectation.collapseIsometries();
  collapsed = residual_expectation.collapseIsometries();
 }
 //Print final tensor expansions:
 if(TensorNetworkOptimizer::debug > 1){
  if(COLLAPSE_ISOMETRIES){
   std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Collapsed TN functionals:" << std::endl;
   std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Operator expectation expansion:" << std::endl;
   operator_expectation.printIt();
   std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Metrics expectation expansion:" << std::endl;
   metrics_expectation.printIt();
   std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Residual expectation expansion:" << std::endl;
   residual_expectation.printIt();
  }
  std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Derivatives:" << std::endl;
  for(const auto & environment: environments_){
   std::cout << "#DEBUG: Derivative tensor network expansion w.r.t. " << environment.tensor->getName() << std::endl;
   environment.gradient_expansion.printIt();
  }
 }

 //Define the metrics scaling function:
 auto scale_metrics = [this](TensorExpansion & expansion,
                             std::complex<double> old_eigenvalue,
                             std::complex<double> new_eigenvalue){
  for(auto component = expansion.begin(); component != expansion.end(); ++component){
   if(component->network->getName().find("MetrExpect") == 0) component->coefficient *= (new_eigenvalue / old_eigenvalue);
  }
  return;
 };

 //Tensor optimization procedure:
 bool converged = environments_.empty();
 if(!converged){
  //Create a scalar tensor:
  auto scalar_norm = makeSharedTensor("_scalar_norm");
  bool done = createTensorSync(scalar_norm,environments_[0].tensor->getElementType()); assert(done);
  //Iterate:
  unsigned int iteration = 0;
  while((!converged) && (iteration < max_iterations_)){
   if(TensorNetworkOptimizer::debug > 0)
    std::cout << "#DEBUG(exatn::TensorNetworkOptimizer)["
              << std::fixed << std::setprecision(6) << exatn::Timer::timeInSecHR(numericalServer->getTimeStampStart())
              << "]: Iteration " << iteration << std::endl;
   converged = true;
   double max_convergence = 0.0;
   average_expect_val_ = std::complex<double>{0.0,0.0};
   for(auto & environment: environments_){
    //Create the gradient tensors:
    done = createTensorSync(environment.gradient,environment.tensor->getElementType()); assert(done);
    done = createTensorSync(environment.gradient_aux,environment.tensor->getElementType()); assert(done);
    //Microiterations:
    double local_convergence = 0.0;
    for(unsigned int micro_iteration = 0; micro_iteration < micro_iterations_; ++micro_iteration){
     //Compute the metrics expectation value w.r.t. the optimized tensor (real positive definite):
     done = initTensorSync("_scalar_norm",0.0); assert(done);
     done = evaluateSync(process_group,metrics_expectation,scalar_norm,num_procs); assert(done);
     double tens_norm = 0.0;
     done = computeNorm1Sync("_scalar_norm",tens_norm); assert(done);
     assert(tens_norm > 0.0);
     //Compute the operator expectation value w.r.t. the optimized tensor (real):
     done = initTensorSync("_scalar_norm",0.0); assert(done);
     done = evaluateSync(process_group,operator_expectation,scalar_norm,num_procs); assert(done);
     std::complex<double> expect_val{0.0,0.0};
     switch(scalar_norm->getElementType()){
      case TensorElementType::REAL32:
       expect_val = {exatn::getLocalTensor("_scalar_norm")->getSliceView<float>()[std::initializer_list<int>{}],0.0f};
       break;
      case TensorElementType::REAL64:
       expect_val = {exatn::getLocalTensor("_scalar_norm")->getSliceView<double>()[std::initializer_list<int>{}],0.0};
       break;
      case TensorElementType::COMPLEX32:
       expect_val = exatn::getLocalTensor("_scalar_norm")->getSliceView<std::complex<float>>()[std::initializer_list<int>{}];
       break;
      case TensorElementType::COMPLEX64:
       expect_val = exatn::getLocalTensor("_scalar_norm")->getSliceView<std::complex<double>>()[std::initializer_list<int>{}];
       break;
      default:
       assert(false);
     }
     expect_val /= std::complex<double>{tens_norm,0.0}; //average operator expectation value w.r.t. the optimized tensor
     if(micro_iteration == (micro_iterations_ - 1)) average_expect_val_ += expect_val;
     if(TensorNetworkOptimizer::debug > 1) std::cout << " Operator expectation value w.r.t. " << environment.tensor->getName()
                                                     << " = " << std::scientific << expect_val << std::endl;
     //Normalize the optimized tensor w.r.t. metrics:
     if(!(environment.tensor->hasIsometries())){
      done = scaleTensorSync(environment.tensor->getName(),1.0/std::sqrt(tens_norm)); assert(done);
     }
     //Update the average operator expectation value in the gradient expansion:
     if(TensorNetworkOptimizer::debug > 2){
      std::cout << " Old gradient expansion coefficients:\n";
      environment.gradient_expansion.printCoefficients();
     }
     scale_metrics(environment.gradient_expansion,environment.expect_value,expect_val);
     if(TensorNetworkOptimizer::debug > 2){
      std::cout << " New gradient expansion coefficients:\n";
      environment.gradient_expansion.printCoefficients();
     }
     //Initialize the gradient tensor to zero:
     done = initTensorSync(environment.gradient->getName(),0.0); assert(done);
     //Evaluate the gradient tensor expansion:
     done = evaluateSync(process_group,environment.gradient_expansion,environment.gradient,num_procs); assert(done);
     //Compute the norm of the gradient tensor:
     double grad_norm = 0.0;
     done = computeNorm2Sync(environment.gradient->getName(),grad_norm); assert(done);
     if(TensorNetworkOptimizer::debug > 1) std::cout << " Gradient norm w.r.t. " << environment.tensor->getName()
                                                     << " = " << grad_norm << std::endl;
     //Compute the convergence criterion:
     double denom = 0.0;
     done = initTensorSync(environment.gradient_aux->getName(),0.0); assert(done);
     done = evaluateSync(process_group,environment.operator_gradient,environment.gradient_aux,num_procs); assert(done);
     tens_norm = 0.0;
     done = computeNorm2Sync(environment.gradient_aux->getName(),tens_norm); assert(done);
     if(TensorNetworkOptimizer::debug > 1) std::cout << environment.tensor->getName()
                                                     << ": |H|x> 2-norm = " << tens_norm;
     denom += tens_norm;
     done = initTensorSync(environment.gradient_aux->getName(),0.0); assert(done);
     done = evaluateSync(process_group,environment.metrics_gradient,environment.gradient_aux,num_procs); assert(done);
     tens_norm = 0.0;
     done = computeNorm2Sync(environment.gradient_aux->getName(),tens_norm); assert(done);
     if(TensorNetworkOptimizer::debug > 1) std::cout << "; |S|x> 2-norm = " << tens_norm
                                                     << "; Absolute eigenvalue = " << std::abs(expect_val) << std::endl;
     denom += std::abs(expect_val) * tens_norm;
     local_convergence = grad_norm / denom;
     if(TensorNetworkOptimizer::debug > 1) std::cout << " Convergence w.r.t. " << environment.tensor->getName()
                                                     << " = " << grad_norm << " / " << denom
                                                     << " = " << local_convergence << std::endl;
     if(local_convergence > tolerance_){
      if(micro_iteration == (micro_iterations_ - 1)) converged = false;
     }
     //Compute the optimal step size:
     if(TensorNetworkOptimizer::debug > 2){
      std::cout << " Old hessian expansion coefficients:\n";
      environment.hessian_expansion.printCoefficients();
     }
     scale_metrics(environment.hessian_expansion,environment.expect_value,expect_val);
     if(TensorNetworkOptimizer::debug > 2){
      std::cout << " New hessian expansion coefficients:\n";
      environment.hessian_expansion.printCoefficients();
     }
     done = initTensorSync("_scalar_norm",0.0); assert(done);
     done = evaluateSync(process_group,environment.hessian_expansion,scalar_norm,num_procs); assert(done);
     denom = 0.0;
     switch(scalar_norm->getElementType()){
      case TensorElementType::REAL32:
       denom = exatn::getLocalTensor("_scalar_norm")->getSliceView<float>()[std::initializer_list<int>{}];
       break;
      case TensorElementType::REAL64:
       denom = exatn::getLocalTensor("_scalar_norm")->getSliceView<double>()[std::initializer_list<int>{}];
       break;
      case TensorElementType::COMPLEX32:
       denom = exatn::getLocalTensor("_scalar_norm")->getSliceView<std::complex<float>>()[std::initializer_list<int>{}].real();
       break;
      case TensorElementType::COMPLEX64:
       denom = exatn::getLocalTensor("_scalar_norm")->getSliceView<std::complex<double>>()[std::initializer_list<int>{}].real();
       break;
      default:
       assert(false);
     }
     if(TensorNetworkOptimizer::debug > 1) std::cout << " Step size for " << environment.tensor->getName() << " = "
                                                     << (grad_norm * grad_norm) << " / " << denom << " = "
                                                     << (grad_norm * grad_norm / denom) << std::endl;
     denom = std::abs(denom);
     if(denom > MIN_ACCEPTABLE_DENOM){
      epsilon_ = grad_norm * grad_norm / denom;
      if(TensorNetworkOptimizer::debug > 1) std::cout << " Optimal step size = " << epsilon_
                                                      << ": Denominator = " << denom << std::endl;
     }else{
      epsilon_ = DEFAULT_LEARN_RATE;
      if(TensorNetworkOptimizer::debug > 1) std::cout << " Optimal step size = " << epsilon_ << std::endl;
     }
     //Update the optimized tensor:
     std::string add_pattern;
     done = generate_addition_pattern(environment.tensor->getRank(),add_pattern,false,
                                      environment.tensor->getName(),environment.gradient->getName()); assert(done);
     done = addTensorsSync(add_pattern,-epsilon_); assert(done);
     if(!(environment.tensor->hasIsometries())){
      if(NORMALIZE_WITH_METRICS){
       //Normalize the optimized tensor w.r.t. metrics:
       done = initTensorSync("_scalar_norm",0.0); assert(done);
       done = evaluateSync(process_group,metrics_expectation,scalar_norm,num_procs); assert(done);
       tens_norm = 0.0;
       done = computeNorm1Sync("_scalar_norm",tens_norm); assert(done);
       if(TensorNetworkOptimizer::debug > 1) std::cout << " Metrical tensor norm before normalization = "
                                                       << std::sqrt(tens_norm) << std::endl;
       assert(tens_norm > 0.0);
       done = scaleTensorSync(environment.tensor->getName(),1.0/std::sqrt(tens_norm)); assert(done);
      }else{
       //Normalize the optimized tensor with unity metrics:
       tens_norm = 0.0;
       done = computeNorm2Sync(environment.tensor->getName(),tens_norm); assert(done);
       if(TensorNetworkOptimizer::debug > 1) std::cout << " Regular tensor norm before normalization = "
                                                       << tens_norm << std::endl;
       assert(tens_norm > 0.0);
       done = scaleTensorSync(environment.tensor->getName(),1.0/tens_norm); assert(done);
      }
     }
     //Update the old expectation value:
     environment.expect_value = expect_val;
     if(local_convergence <= tolerance_) break;
    } //micro-iterations
    //Update the convergence residual:
    max_convergence = std::max(max_convergence,local_convergence);
    //Destroy the gradient tensors:
    done = destroyTensorSync(environment.gradient_aux->getName()); assert(done);
    done = destroyTensorSync(environment.gradient->getName()); assert(done);
   }
   average_expect_val_ /= static_cast<double>(environments_.size());
   if(TensorNetworkOptimizer::debug > 0){
    std::cout << "Average expectation value = " << average_expect_val_
              << "; Max convergence residual = " << max_convergence << std::endl;
   }
   ++iteration;
  }
  //Destroy the scalar tensor:
  done = destroyTensorSync("_scalar_norm"); assert(done);
 }

 //Deactivate caching of optimal tensor contraction sequences:
 if(!con_seq_caching) deactivateContrSeqCaching();
 return converged;
}


bool TensorNetworkOptimizer::optimize_tr(const ProcessGroup & process_group)
{
 constexpr double MIN_ACCEPTABLE_DENOM = 1e-13; //minimally acceptable denominator in optimal step size determination
 constexpr bool COLLAPSE_ISOMETRIES = true;     //enables collapsing isometries in all tensor networks

 unsigned int local_rank; //local process rank within the process group
 if(!process_group.rankIsIn(exatn::getProcessRank(),&local_rank)) return true; //process is not in the group: Do nothing
 unsigned int num_procs = 1;
 if(parallel_) num_procs = process_group.getSize();

 if(TensorNetworkOptimizer::focus >= 0){
  if(getProcessRank() != TensorNetworkOptimizer::focus) TensorNetworkOptimizer::debug = 0;
 }

 if(TensorNetworkOptimizer::debug > 0){
  std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Tensor network operator:" << std::endl;
  tensor_operator_->printIt();
  std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Tensor network vector:" << std::endl;
  vector_expansion_->printIt();
 }

 //Activate caching of optimal tensor contraction sequences:
 bool con_seq_caching = queryContrSeqCaching();
 if(!con_seq_caching) activateContrSeqCaching();

 //Construct the operator expectation expansion:
 // <vector|operator|vector>
 TensorExpansion bra_vector_expansion(*vector_expansion_);
 bra_vector_expansion.conjugate();
 bra_vector_expansion.rename(vector_expansion_->getName()+"Bra");
 TensorExpansion operator_expectation(bra_vector_expansion,*vector_expansion_,*tensor_operator_);
 operator_expectation.rename("OperatorExpectation");
 for(auto net = operator_expectation.begin(); net != operator_expectation.end(); ++net){
  net->network->rename("OperExpect" + std::to_string(std::distance(operator_expectation.begin(),net)));
 }
 if(TensorNetworkOptimizer::debug > 1){
  std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Operator expectation expansion:" << std::endl;
  operator_expectation.printIt();
 }

 //Prepare derivative environments for all optimizable tensors in the vector expansion:
 environments_.clear();
 std::unordered_set<std::string> tensor_names;
 // Loop over the tensor networks constituting the tensor network vector expansion:
 for(auto network = vector_expansion_->cbegin(); network != vector_expansion_->cend(); ++network){
  // Loop over the optimizable tensors inside the current tensor network:
  for(auto tensor_conn = network->network->begin(); tensor_conn != network->network->end(); ++tensor_conn){
   const auto & tensor = tensor_conn->second;
   if(tensor.isOptimizable()){ //gradient w.r.t. an optimizable tensor inside the tensor network vector expansion
    make_sure(tensor.hasIsometries(),"#ERROR(exatn::TensorNetworkOptimizer): Unable to optimize non-isometric tensors in trace minimization!");
    auto res = tensor_names.emplace(tensor.getName());
    if(res.second){ //prepare derivative environment only once for each unique tensor name
     auto gradient_tensor = std::make_shared<Tensor>("_g"+tensor.getName(),tensor.getShape(),tensor.getSignature());
     std::vector<TensorLeg> iso_self_pattern;
     auto overlap_tensor = std::make_shared<Tensor>("_s"+tensor.getName(),*(tensor.getTensor()),iso_self_pattern,0);
     environments_.emplace_back(Environment{tensor.getTensor(),                              //optimizable tensor (isometric)
                                            gradient_tensor,                                 //gradient tensor (non-isometric)
                                            std::make_shared<Tensor>("_h"+tensor.getName(),  //auxiliary gradient tensor (non-isometric)
                                                                     tensor.getShape(),
                                                                     tensor.getSignature()),
                                            overlap_tensor,                                  //gradient-tensor overlap
                                            TensorExpansion(operator_expectation,tensor.getName(),true), // |operator|tensor>
                                            TensorExpansion(operator_expectation,tensor.getName(),true), // |operator|tensor>
                                            TensorExpansion(), //no metrics
                                            TensorExpansion(operator_expectation,tensor.getTensor(),gradient_tensor), // <gradient|operator|gradient>
                                            {1.0,0.0}});
     if(COLLAPSE_ISOMETRIES){
      auto collapsed = environments_.back().gradient_expansion.collapseIsometries();
      collapsed = environments_.back().operator_gradient.collapseIsometries();
      collapsed = environments_.back().hessian_expansion.collapseIsometries();
     }
    }
   }
  }
 }
 //Collapse isometries in the original TN functionals:
 if(COLLAPSE_ISOMETRIES){
  auto collapsed = operator_expectation.collapseIsometries();
 }
 //Print final tensor expansions:
 if(TensorNetworkOptimizer::debug > 1){
  if(COLLAPSE_ISOMETRIES){
   std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Collapsed TN functionals:" << std::endl;
   std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Operator expectation expansion:" << std::endl;
   operator_expectation.printIt();
  }
  std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Derivatives:" << std::endl;
  for(const auto & environment: environments_){
   std::cout << "#DEBUG: Derivative tensor network expansion w.r.t. " << environment.tensor->getName() << std::endl;
   environment.gradient_expansion.printIt();
  }
 }

 //Tensor optimization procedure:
 bool converged = environments_.empty();
 if(!converged){
  //Create a scalar tensor:
  auto scalar_norm = makeSharedTensor("_scalar_norm");
  bool done = createTensorSync(scalar_norm,environments_[0].tensor->getElementType()); assert(done);
  //Iterate:
  unsigned int iteration = 0;
  while((!converged) && (iteration < max_iterations_)){
   if(TensorNetworkOptimizer::debug > 0)
    std::cout << "#DEBUG(exatn::TensorNetworkOptimizer)["
              << std::fixed << std::setprecision(6) << exatn::Timer::timeInSecHR(numericalServer->getTimeStampStart())
              << "]: Iteration " << iteration << std::endl;
   converged = true;
   double max_convergence = 0.0;
   average_expect_val_ = std::complex<double>{0.0,0.0};
   for(auto & environment: environments_){
    //Create the gradient tensors:
    done = createTensorSync(environment.gradient,environment.tensor->getElementType()); assert(done);
    done = createTensorSync(environment.gradient_aux,environment.tensor->getElementType()); assert(done);
    //Microiterations:
    double grad_norm = 0.0;
    for(unsigned int micro_iteration = 0; micro_iteration < micro_iterations_; ++micro_iteration){
     //Compute the operator expectation value w.r.t. the optimized tensor (real):
     done = initTensorSync("_scalar_norm",0.0); assert(done);
     done = evaluateSync(process_group,operator_expectation,scalar_norm,num_procs); assert(done);
     std::complex<double> expect_val{0.0,0.0};
     switch(scalar_norm->getElementType()){
      case TensorElementType::REAL32:
       expect_val = {exatn::getLocalTensor("_scalar_norm")->getSliceView<float>()[std::initializer_list<int>{}],0.0f};
       break;
      case TensorElementType::REAL64:
       expect_val = {exatn::getLocalTensor("_scalar_norm")->getSliceView<double>()[std::initializer_list<int>{}],0.0};
       break;
      case TensorElementType::COMPLEX32:
       expect_val = exatn::getLocalTensor("_scalar_norm")->getSliceView<std::complex<float>>()[std::initializer_list<int>{}];
       break;
      case TensorElementType::COMPLEX64:
       expect_val = exatn::getLocalTensor("_scalar_norm")->getSliceView<std::complex<double>>()[std::initializer_list<int>{}];
       break;
      default:
       assert(false);
     }
     if(micro_iteration == (micro_iterations_ - 1)) average_expect_val_ += expect_val;
     if(TensorNetworkOptimizer::debug > 1) std::cout << " Operator expectation value w.r.t. " << environment.tensor->getName()
                                                     << " = " << std::scientific << expect_val << std::endl;
     //Initialize the gradient tensor to zero:
     done = initTensorSync(environment.gradient->getName(),0.0); assert(done);
     //Evaluate the gradient tensor expansion:
     done = evaluateSync(process_group,environment.gradient_expansion,environment.gradient,num_procs); assert(done);
     //Compute the norm of the gradient tensor:
     grad_norm = 0.0;
     done = computeNorm2Sync(environment.gradient->getName(),grad_norm); assert(done);
     //Compute the norm of the current tensor (debug):
     double tens_norm = 0.0;
     done = computeNorm2Sync(environment.tensor->getName(),tens_norm); assert(done);
     if(TensorNetworkOptimizer::debug > 1) std::cout << " Gradient norm w.r.t. " << environment.tensor->getName()
                                                     << " = " << grad_norm << "; Tensor norm = " << tens_norm << std::endl;
     if(grad_norm > tolerance_){
      if(micro_iteration == (micro_iterations_ - 1)) converged = false;
     }
     //Compute the step size:
     epsilon_ = DEFAULT_LEARN_RATE;
     if(TensorNetworkOptimizer::debug > 1) std::cout << " Optimal step size = " << epsilon_ << std::endl;
     //Update the optimized tensor:
     std::string add_pattern;
     done = generate_addition_pattern(environment.tensor->getRank(),add_pattern,false,
                                      environment.tensor->getName(),environment.gradient->getName()); assert(done);
     done = addTensorsSync(add_pattern,-epsilon_); assert(done);
     //Update the old expectation value:
     environment.expect_value = expect_val;
     if(grad_norm <= tolerance_) break;
    } //micro-iterations
    //Update the convergence residual:
    max_convergence = std::max(max_convergence,grad_norm);
    //Destroy the gradient tensors:
    done = destroyTensorSync(environment.gradient_aux->getName()); assert(done);
    done = destroyTensorSync(environment.gradient->getName()); assert(done);
   } //tensors
   average_expect_val_ /= static_cast<double>(environments_.size());
   if(TensorNetworkOptimizer::debug > 0){
    std::cout << "Average expectation value = " << average_expect_val_
              << "; Max convergence residual = " << max_convergence << std::endl;
   }
   ++iteration;
  }
  //Destroy the scalar tensor:
  done = destroyTensorSync("_scalar_norm"); assert(done);
 }

 //Deactivate caching of optimal tensor contraction sequences:
 if(!con_seq_caching) deactivateContrSeqCaching();
 return converged;
}


void TensorNetworkOptimizer::computeInitialGuess(const ProcessGroup & process_group,
                                                 bool highest,
                                                 unsigned int guess_dim)
{
 assert(tensor_operator_ && vector_expansion_);
 assert(guess_dim > 1);

 unsigned int num_procs = 1;
 if(parallel_) num_procs = process_group.getSize();

 if(TensorNetworkOptimizer::focus >= 0){
  if(getProcessRank() != TensorNetworkOptimizer::focus) TensorNetworkOptimizer::debug = 0;
 }

 bool success = true;
 //Generate a random non-orthogonal tensor network basis:
 auto tn_builder = exatn::getTensorNetworkBuilder("MPS");
 success = tn_builder->setParameter("max_bond_dim",DEFAULT_GUESS_MAX_BOND_DIM); assert(success);
 auto ket_tensor = vector_expansion_->getSpaceTensor();
 const auto elem_type = vector_expansion_->cbegin()->network->getTensorElementType();
 assert(elem_type != TensorElementType::VOID);
 std::vector<std::shared_ptr<TensorExpansion>> basis_ket(guess_dim);
 std::vector<std::shared_ptr<TensorExpansion>> basis_bra(guess_dim);
 success = exatn::sync(process_group); assert(success);
 for(unsigned int i = 0; i < guess_dim; ++i){
  basis_ket[i] = makeSharedTensorExpansion("_BasisVector"+std::to_string(i));
  auto basis_vector = makeSharedTensorNetwork("_VectorNet"+std::to_string(i),ket_tensor,*tn_builder,false);
  success = basis_ket[i]->appendComponent(basis_vector,std::complex<double>{1.0,0.0}); assert(success);
  success = exatn::createTensors(process_group,*basis_vector,elem_type); assert(success);
  success = exatn::initTensorsRnd(*basis_vector); assert(success);
 }
 success = exatn::sync(process_group); assert(success);
 //Normalize the non-orthogonal tensor network basis:
 for(unsigned int i = 0; i < guess_dim; ++i){
  success = normalizeNorm2Sync(process_group,*(basis_ket[i]),1.0); assert(success);
  basis_bra[i] = makeSharedTensorExpansion(*(basis_ket[i]));
  basis_bra[i]->conjugate();
  basis_bra[i]->rename("_ConjBasisVector"+std::to_string(i));
 }
 success = exatn::sync(process_group); assert(success);
 //Build the operator matrix:
 std::vector<std::complex<double>> oper_matrix(guess_dim*guess_dim);
 std::vector<std::shared_ptr<Tensor>> oper_scalar(guess_dim*guess_dim);
 std::vector<std::shared_ptr<TensorExpansion>> oper_elems(guess_dim*guess_dim);
 for(unsigned int j = 0; j < guess_dim; ++j){
  for(unsigned int i = 0; i < guess_dim; ++i){
   oper_elems[j*guess_dim + i] = makeSharedTensorExpansion(*(basis_ket[j]),*(basis_bra[i]),*tensor_operator_);
   oper_scalar[j*guess_dim + i] = makeSharedTensor("_");
   oper_scalar[j*guess_dim + i]->rename("_OperScalarElem_"+std::to_string(i)+"_"+std::to_string(j));
   success = exatn::createTensor(process_group,oper_scalar[j*guess_dim + i],elem_type); assert(success);
   success = exatn::initTensor(oper_scalar[j*guess_dim + i]->getName(),0.0); assert(success);
   success = exatn::evaluate(process_group,*(oper_elems[j*guess_dim + i]),oper_scalar[j*guess_dim + i],num_procs); assert(success);
  }
 }
 //Build the metric matrix:
 std::vector<std::complex<double>> metr_matrix(guess_dim*guess_dim);
 std::vector<std::shared_ptr<Tensor>> metr_scalar(guess_dim*guess_dim);
 std::vector<std::shared_ptr<TensorExpansion>> metr_elems(guess_dim*guess_dim);
 for(unsigned int j = 0; j < guess_dim; ++j){
  for(unsigned int i = 0; i < guess_dim; ++i){
   metr_elems[j*guess_dim + i] = makeSharedTensorExpansion(*(basis_ket[j]),*(basis_bra[i]));
   metr_scalar[j*guess_dim + i] = makeSharedTensor("_");
   metr_scalar[j*guess_dim + i]->rename("_MetrScalarElem_"+std::to_string(i)+"_"+std::to_string(j));
   success = exatn::createTensor(process_group,metr_scalar[j*guess_dim + i],elem_type); assert(success);
   success = exatn::initTensor(metr_scalar[j*guess_dim + i]->getName(),0.0); assert(success);
   success = exatn::evaluate(process_group,*(metr_elems[j*guess_dim + i]),metr_scalar[j*guess_dim + i],num_procs); assert(success);
  }
 }
 success = exatn::sync(process_group); assert(success);
 for(unsigned int j = 0; j < guess_dim; ++j){
  for(unsigned int i = 0; i < guess_dim; ++i){
   const auto & oper_name = oper_scalar[j*guess_dim + i]->getName();
   const auto & metr_name = metr_scalar[j*guess_dim + i]->getName();
   switch(elem_type){
    case TensorElementType::REAL32:
     oper_matrix[j*guess_dim + i] = std::complex<double>(
      exatn::getLocalTensor(oper_name)->getSliceView<float>()[std::initializer_list<int>{}], 0.0);
     metr_matrix[j*guess_dim + i] = std::complex<double>(
      exatn::getLocalTensor(metr_name)->getSliceView<float>()[std::initializer_list<int>{}], 0.0);
     break;
    case TensorElementType::REAL64:
     oper_matrix[j*guess_dim + i] = std::complex<double>(
      exatn::getLocalTensor(oper_name)->getSliceView<double>()[std::initializer_list<int>{}], 0.0);
     metr_matrix[j*guess_dim + i] = std::complex<double>(
      exatn::getLocalTensor(metr_name)->getSliceView<double>()[std::initializer_list<int>{}], 0.0);
     break;
    case TensorElementType::COMPLEX32:
     oper_matrix[j*guess_dim + i] = std::complex<double>(
      exatn::getLocalTensor(oper_name)->getSliceView<std::complex<float>>()[std::initializer_list<int>{}]);
     metr_matrix[j*guess_dim + i] = std::complex<double>(
      exatn::getLocalTensor(metr_name)->getSliceView<std::complex<float>>()[std::initializer_list<int>{}]);
     break;
    case TensorElementType::COMPLEX64:
     oper_matrix[j*guess_dim + i] = exatn::getLocalTensor(oper_name)->getSliceView<std::complex<double>>()[std::initializer_list<int>{}];
     metr_matrix[j*guess_dim + i] = exatn::getLocalTensor(metr_name)->getSliceView<std::complex<double>>()[std::initializer_list<int>{}];
     break;
    default:
     assert(false);
   }
  }
 }
 //Print matrices (debug):
 if(TensorNetworkOptimizer::debug > 0){
  std::cout << "#DEBUG(exatn::TensorNetworkOptimizer::computeInitialGuess): Operator matrix:\n" << std::scientific;
  for(unsigned int i = 0; i < guess_dim; ++i){
   for(unsigned int j = 0; j < guess_dim; ++j){
    std::cout << " " << oper_matrix[j*guess_dim + i];
   }
   std::cout << std::endl;
  }
  std::cout << "#DEBUG(exatn::TensorNetworkOptimizer::computeInitialGuess): Metric matrix:\n" << std::scientific;
  for(unsigned int i = 0; i < guess_dim; ++i){
   for(unsigned int j = 0; j < guess_dim; ++j){
    std::cout << " " << metr_matrix[j*guess_dim + i];
   }
   std::cout << std::endl;
  }
 }
 //Solve the projected eigen-problem:
 int info = 0;
 const char job_type = 'V';
 const int matrix_dim = guess_dim;
 const int lwork = std::max(2*guess_dim,guess_dim*guess_dim);
 std::vector<std::complex<double>> work_space(lwork);
 std::vector<std::complex<double>> left_vecs(guess_dim*guess_dim,std::complex<double>{0.0,0.0});
 std::vector<std::complex<double>> right_vecs(guess_dim*guess_dim,std::complex<double>{0.0,0.0});
 std::vector<std::complex<double>> alpha(guess_dim,std::complex<double>{0.0,0.0});
 std::vector<std::complex<double>> beta(guess_dim,std::complex<double>{0.0,0.0});
 std::vector<double> rwork_space(guess_dim*8);
 zggev_(&job_type,&job_type,&matrix_dim,
        (void*)oper_matrix.data(),&matrix_dim,
        (void*)metr_matrix.data(),&matrix_dim,
        (void*)alpha.data(),(void*)beta.data(),
        (void*)left_vecs.data(),&matrix_dim,
        (void*)right_vecs.data(),&matrix_dim,
        (void*)work_space.data(),&lwork,
        (void*)rwork_space.data(),&info);
 if(info == 0){
  //Print eigenvalues (debug):
  if(TensorNetworkOptimizer::debug > 0){
   std::cout << "#DEBUG(exatn::TensorNetworkOptimizer::computeInitialGuess): Eigenvalues:\n" << std::scientific;
   for(unsigned int i = 0; i < guess_dim; ++i){
    std::cout << alpha[i] << " / " << beta[i] << " = ";
    if(std::abs(beta[i]) != 0.0) std::cout << (alpha[i]/beta[i]) << std::endl;
   }
  }
  //Find the min/max eigenvalue:
  int min_entry = 0, max_entry = 0;
  for(int i = 0; i < matrix_dim; ++i){
   if(std::abs(beta[i]) != 0.0){
    const auto lambda = alpha[i] / beta[i];
    if(std::abs(beta[min_entry]) != 0.0){
     const auto min_val = alpha[min_entry] / beta[min_entry];
     if(lambda.real() < min_val.real()) min_entry = i;
    }else{
     min_entry = i;
    }
    if(std::abs(beta[max_entry]) != 0.0){
     const auto max_val = alpha[max_entry] / beta[max_entry];
     if(lambda.real() > max_val.real()) max_entry = i;
    }else{
     max_entry = i;
    }
   }
  }
  //Generate the target tensor eigen-expansion:
  auto target_root = min_entry;
  if(highest) target_root = max_entry;
  auto root_expansion = makeSharedTensorExpansion("_RootExpansion");
  for(int i = 0; i < matrix_dim; ++i){
   const auto coef = right_vecs[target_root*matrix_dim + i];
   for(auto net = basis_ket[i]->begin(); net != basis_ket[i]->end(); ++net){
    success = root_expansion->appendComponent(net->network,coef*(net->coefficient)); assert(success);
   }
  }
  success = normalizeNorm2Sync(process_group,*root_expansion,1.0); assert(success);
  //Reconstruct the eigen-expansion as an initial guess:
  vector_expansion_->conjugate();
  TensorNetworkReconstructor::resetDebugLevel(1,0); //debug
  TensorNetworkReconstructor reconstructor(root_expansion,vector_expansion_,DEFAULT_GUESS_TOLERANCE);
  reconstructor.resetMaxIterations(10);
  success = exatn::sync(process_group); assert(success);
  double residual_norm, fidelity;
  bool reconstructed = reconstructor.reconstruct(process_group,&residual_norm,&fidelity);
  success = exatn::sync(process_group); assert(success);
  vector_expansion_->conjugate();
  //success = balanceNormalizeNorm2Sync(process_group,*vector_expansion_,1.0,1.0,true); assert(success); //debug
 }
 //Destroy temporaries:
 for(unsigned int j = 0; j < guess_dim; ++j){
  success = exatn::destroyTensors(*(basis_ket[j]->begin()->network)); assert(success);
  for(unsigned int i = 0; i < guess_dim; ++i){
   success = exatn::destroyTensor(oper_scalar[j*guess_dim + i]->getName()); assert(success);
   success = exatn::destroyTensor(metr_scalar[j*guess_dim + i]->getName()); assert(success);
  }
 }
 success = exatn::sync(process_group); assert(success);
 return;
}


void TensorNetworkOptimizer::enableParallelization(bool parallel)
{
 parallel_ = parallel;
 return;
}


void TensorNetworkOptimizer::resetDebugLevel(unsigned int level, int focus_process)
{
 TensorNetworkOptimizer::debug = level;
 TensorNetworkOptimizer::focus = focus_process;
 return;
}

} //namespace exatn
