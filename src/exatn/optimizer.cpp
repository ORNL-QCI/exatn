/** ExaTN:: Variational optimizer of a closed symmetric tensor network expansion functional
REVISION: 2021/01/21

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "optimizer.hpp"

#include <talshxx.hpp>

#include <unordered_set>

namespace exatn{

unsigned int TensorNetworkOptimizer::debug{0};


TensorNetworkOptimizer::TensorNetworkOptimizer(std::shared_ptr<TensorOperator> tensor_operator,
                                               std::shared_ptr<TensorExpansion> vector_expansion,
                                               double tolerance):
 tensor_operator_(tensor_operator), vector_expansion_(vector_expansion),
 max_iterations_(DEFAULT_MAX_ITERATIONS), epsilon_(DEFAULT_LEARN_RATE), tolerance_(tolerance)
{
 if(!vector_expansion_->isKet()){
  std::cout << "#ERROR(exatn:TensorNetworkOptimizer): The tensor network vector expansion must be a ket!"
            << std::endl << std::flush;
  assert(false);
 }
 if(tensor_operator_->getKetRank() != tensor_operator_->getBraRank()){
  std::cout << "#ERROR(exatn:TensorNetworkOptimizer): Tensor operator is not rank-symmetric!"
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


std::shared_ptr<TensorExpansion> TensorNetworkOptimizer::getSolution() const
{
 return vector_expansion_;
}


bool TensorNetworkOptimizer::optimize()
{
 return optimize(exatn::getDefaultProcessGroup());
}

bool TensorNetworkOptimizer::optimize(const ProcessGroup & process_group)
{
 unsigned int local_rank; //local process rank within the process group
 if(!process_group.rankIsIn(exatn::getProcessRank(),&local_rank)) return true; //process is not in the group: Do nothing

 bool success = true;

 if(TensorNetworkOptimizer::debug){
  std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Tensor network operator:" << std::endl;
  tensor_operator_->printIt();
  std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Tensor network vector:" << std::endl;
  vector_expansion_->printIt();
 }

 //Construct the operator expectation expansion:
 // <vector|operator|vector>
 TensorExpansion bra_vector_expansion(*vector_expansion_);
 bra_vector_expansion.conjugate();
 bra_vector_expansion.rename(vector_expansion_->getName()+"Bra");
 TensorExpansion operator_expectation(bra_vector_expansion,*vector_expansion_,*tensor_operator_);
 operator_expectation.rename("OperatorExpectation");
 if(TensorNetworkOptimizer::debug){
  std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Operator expectation expansion:" << std::endl;
  operator_expectation.printIt();
 }

 //Construct the metrics expectation expansion:
 // <vector|vector>
 TensorExpansion metrics_expectation(bra_vector_expansion,*vector_expansion_);
 metrics_expectation.rename("MetricsExpectation");
 if(TensorNetworkOptimizer::debug){
  std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Metrics expectation expansion:" << std::endl;
  metrics_expectation.printIt();
 }

 //Construct the residual expectation expansion:
 // <vector|operator|vector> - <vector|vector>
 TensorExpansion residual_expectation;
 success = residual_expectation.appendExpansion(operator_expectation,{1.0,0.0}); assert(success);
 success = residual_expectation.appendExpansion(metrics_expectation,{-1.0,0.0}); assert(success);
 residual_expectation.rename("ResidualExpectation");
 if(TensorNetworkOptimizer::debug){
  std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Residual expectation expansion:" << std::endl;
  residual_expectation.printIt();
 }

 //Prepare derivative environments for all optimizable tensors in the vector expansion:
 std::unordered_set<std::string> tensor_names;
 // Loop over the tensor networks constituting the tensor network vector expansion:
 for(auto network = vector_expansion_->cbegin(); network != vector_expansion_->cend(); ++network){
  // Loop over the optimizable tensors inside the current tensor network:
  for(auto tensor_conn = network->network_->begin(); tensor_conn != network->network_->end(); ++tensor_conn){
   const auto & tensor = tensor_conn->second;
   if(tensor.isOptimizable()){ //gradient w.r.t. an optimizable tensor inside the tensor network vector expansion
    auto res = tensor_names.emplace(tensor.getName());
    if(res.second){ //prepare derivative environment only once for each unique tensor name
     auto gradient_tensor = std::make_shared<Tensor>("_g"+tensor.getName(),tensor.getShape(),tensor.getSignature());
     environments_.emplace_back(Environment{tensor.getTensor(),                              //optimizable tensor
                                            gradient_tensor,                                 //gradient tensor
                                            std::make_shared<Tensor>("_h"+tensor.getName(),  //auxiliary gradient tensor
                                                                     tensor.getShape(),
                                                                     tensor.getSignature()),
                                            TensorExpansion(residual_expectation,tensor.getName(),true), // |operator|tensor> - |metrics|tensor>
                                            TensorExpansion(operator_expectation,tensor.getName(),true), // |operator|tensor>
                                            TensorExpansion(metrics_expectation,tensor.getName(),true),  // |metrics|tensor>
                                            TensorExpansion(residual_expectation,tensor.getTensor(),gradient_tensor)}); // <gradient|operator|gradient> - <gradient|metrics|gradient>
    }
   }
  }
 }
 if(TensorNetworkOptimizer::debug){
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
   if(TensorNetworkOptimizer::debug)
    std::cout << "#DEBUG(exatn::TensorNetworkOptimizer): Iteration " << iteration << std::endl;
   converged = true;
   for(auto & environment: environments_){
    //Normalize the optimized tensor w.r.t. metrics:
    done = initTensorSync("_scalar_norm",0.0); assert(done);
    done = evaluateSync(process_group,metrics_expectation,scalar_norm); assert(done);
    double tens_norm = 0.0;
    done = computeNorm1Sync("_scalar_norm",tens_norm); assert(done);
    tens_norm = std::sqrt(tens_norm);
    done = scaleTensorSync(environment.tensor->getName(),1.0/tens_norm); assert(done);
    //Compute the operator expectation value w.r.t. the optimized tensor:
    done = initTensorSync("_scalar_norm",0.0); assert(done);
    done = evaluateSync(process_group,operator_expectation,scalar_norm); assert(done);
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
    if(TensorNetworkOptimizer::debug) std::cout << " Operator expectation value w.r.t. " << environment.tensor->getName()
                                                << " = " << expect_val << std::endl;
    //Update the expectation value in the gradient expansion:
    environment.gradient_expansion[1].coefficient_ = -expect_val;
    //Create the gradient tensor:
    done = createTensorSync(environment.gradient,environment.tensor->getElementType()); assert(done);
    //Initialize the gradient tensor to zero:
    done = initTensorSync(environment.gradient->getName(),0.0); assert(done);
    //Evaluate the gradient tensor expansion:
    done = evaluateSync(process_group,environment.gradient_expansion,environment.gradient); assert(done);
    //Compute the norm of the gradient tensor:
    double grad_norm = 0.0;
    done = computeNorm2Sync(environment.gradient->getName(),grad_norm); assert(done);
    if(TensorNetworkOptimizer::debug) std::cout << " Gradient norm w.r.t. " << environment.tensor->getName()
                                                << " = " << grad_norm << std::endl;
    //Compute the convergence criterion:
    double denom = 0.0;
    done = createTensorSync(environment.gradient_aux,environment.tensor->getElementType()); assert(done);
    done = initTensorSync(environment.gradient_aux->getName(),0.0); assert(done);
    done = evaluateSync(process_group,environment.operator_gradient,environment.gradient_aux); assert(done);
    tens_norm = 0.0;
    done = computeNorm2Sync(environment.gradient_aux->getName(),tens_norm); assert(done);
    denom += tens_norm;
    done = initTensorSync(environment.gradient_aux->getName(),0.0); assert(done);
    done = evaluateSync(process_group,environment.metrics_gradient,environment.gradient_aux); assert(done);
    tens_norm = 0.0;
    done = computeNorm2Sync(environment.gradient_aux->getName(),tens_norm); assert(done);
    denom += std::abs(expect_val) * tens_norm;
    if(TensorNetworkOptimizer::debug) std::cout << " Convergence w.r.t. " << environment.tensor->getName()
                                                << " = " << (grad_norm / denom) << std::endl;
    if(grad_norm / denom > tolerance_){
     converged = false;
     //Compute the optimal step size:
     done = initTensorSync("_scalar_norm",0.0); assert(done);
     environment.hessian_expansion[1].coefficient_ = -expect_val;
     done = evaluateSync(process_group,environment.hessian_expansion,scalar_norm); assert(done);
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
     epsilon_ = grad_norm * grad_norm / denom;
     if(TensorNetworkOptimizer::debug) std::cout << " Optimal step size = " << epsilon_ << std::endl;
     //Update the optimized tensor:
     std::string add_pattern;
     done = generate_addition_pattern(environment.tensor->getRank(),add_pattern,true, //`Do I need conjugation here?
                                      environment.tensor->getName(),environment.gradient->getName()); assert(done);
     done = addTensorsSync(add_pattern,-epsilon_); assert(done);
     //Normalize the optimized tensor w.r.t. metrics:
     done = initTensorSync("_scalar_norm",0.0); assert(done);
     done = evaluateSync(process_group,metrics_expectation,scalar_norm); assert(done);
     tens_norm = 0.0;
     done = computeNorm1Sync("_scalar_norm",tens_norm); assert(done);
     tens_norm = std::sqrt(tens_norm);
     done = scaleTensorSync(environment.tensor->getName(),1.0/tens_norm); assert(done);
    }
    //Destroy the gradient tensors:
    done = destroyTensorSync(environment.gradient_aux->getName()); assert(done);
    done = destroyTensorSync(environment.gradient->getName()); assert(done);
   }
   ++iteration;
  }
  //Destroy the scalar tensor:
  done = destroyTensorSync("_scalar_norm"); assert(done);
 }

 return converged;
}


void TensorNetworkOptimizer::resetDebugLevel(unsigned int level)
{
 TensorNetworkOptimizer::debug = level;
 return;
}

} //namespace exatn
