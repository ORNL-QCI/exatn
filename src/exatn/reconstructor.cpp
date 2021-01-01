/** ExaTN:: Reconstructs an approximate tensor network expansion for a given tensor network expansion
REVISION: 2021/01/01

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "reconstructor.hpp"

#include <unordered_set>
#include <string>

namespace exatn{

TensorNetworkReconstructor::TensorNetworkReconstructor(std::shared_ptr<TensorExpansion> expansion,
                                                       std::shared_ptr<TensorExpansion> approximant,
                                                       double tolerance):
 expansion_(expansion), approximant_(approximant), epsilon_(0.1), tolerance_(tolerance), fidelity_(0.0)
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


std::shared_ptr<TensorExpansion> TensorNetworkReconstructor::getSolution(double * fidelity)
{
 if(fidelity_ == 0.0) return std::shared_ptr<TensorExpansion>(nullptr);
 if(fidelity != nullptr) *fidelity = fidelity_;
 return approximant_;
}


bool TensorNetworkReconstructor::reconstruct(double * fidelity)
{
 assert(fidelity != nullptr);

 //Construct the Lagrangian optimization functional (scalar):
 // <approximant|approximant> - <approximant|expansion>
 TensorExpansion approximant_ket(*approximant_,false); // <approximant|
 approximant_ket.conjugate(); // |approximant>
 TensorExpansion overlap(*approximant_,*expansion_); // <approximant|expansion>
 TensorExpansion normalization(*approximant_,approximant_ket); // <approximant|approximant>
 TensorExpansion lagrangian;
 lagrangian.appendExpansion(normalization,{1.0,0.0});
 lagrangian.appendExpansion(overlap,{-1.0,0.0});
 lagrangian.rename("Lagrangian");

 std::cout << "#DEBUG(exatn::TensorNetworkReconstructor): Lagrangian:" << std::endl; //debug
 lagrangian.printIt(); //debug

 //Construct the residual functional (real scalar cost function):
 // <expansion|expansion> + <approximant|approximant> - <approximant|expansion> - <expansion|approximant>
 TensorExpansion expansion_bra(*expansion_,false); // |expansion>
 expansion_bra.conjugate(); // <expansion|
 TensorExpansion input_norm(expansion_bra,*expansion_); // <expansion|expansion>
 TensorExpansion residual;
 residual.appendExpansion(input_norm,{1.0,0.0});
 residual.appendExpansion(normalization,{1.0,0.0});
 residual.appendExpansion(overlap,{-1.0,0.0});
 residual.appendExpansion(TensorExpansion(expansion_bra,approximant_ket),{-1.0,0.0});
 residual.rename("Residual");

 std::cout << "#DEBUG(exatn::TensorNetworkReconstructor): Residual:" << std::endl; //debug
 residual.printIt(); //debug

 //Prepare derivative environments for all optimizable tensors in the approximant:
 std::unordered_set<std::string> tensor_names;
 // Loop over the tensor networks constituting the approximant tensor expansion:
 for(auto network = approximant_->cbegin(); network != approximant_->cend(); ++network){
  // Loop over the optimizable tensors inside the current tensor network:
  for(auto tensor_conn = network->network_->begin(); tensor_conn != network->network_->end(); ++tensor_conn){
   auto tensor = tensor_conn->second;
   if(tensor.isOptimizable()){ //gradient w.r.t. an optimizable tensor inside the approximant tensor expansion
    auto res = tensor_names.emplace(tensor.getName());
    if(res.second){ //prepare derivative environment only once for each unique tensor name
     environments_.emplace_back(Environment{tensor.getTensor(), //optimizable tensor
                                            std::make_shared<Tensor>("_g"+tensor.getName(), //its gradient tensor
                                                                     tensor.getShape(),
                                                                     tensor.getSignature()),
                                            TensorExpansion(lagrangian,tensor.getName(),true)}); //derivative tensor network expansion
    }                                                                                            //w.r.t. conjugated (bra) tensor
   }
  }
 }

 std::cout << "#DEBUG(exatn::TensorNetworkReconstructor): Derivatives:" << std::endl; //debug
 for(const auto & environment: environments_){ //debug
  std::cout << "#DEBUG: Derivative tensor network expansion w.r.t. " << environment.tensor->getName() << std::endl;
  environment.gradient_expansion.printIt();
 }

 //Tensor optimization procedure:
 bool converged = environments_.empty();
 if(!converged){
  //Compute the 2-norm of the input tensor network expansion:
  auto scalar_norm = makeSharedTensor("_scalar_norm");
  bool done = createTensorSync(scalar_norm,environments_[0].tensor->getElementType()); assert(done);
  done = initTensorSync("_scalar_norm",0.0); assert(done);
  done = evaluateSync(input_norm,scalar_norm); assert(done);
  double input_expansion_norm = 0.0;
  done = computeNorm1Sync("_scalar_norm",input_expansion_norm); assert(done);
  std::cout << "#DEBUG(exatn::TensorNetworkReconstructor): 2-norm of the input tensor network expansion = "
            << input_expansion_norm << std::endl; //debug
  while(!converged){
   double max_grad_maxabs = 0.0;
   for(auto & environment: environments_){
    //Create the gradient tensor:
    done = createTensorSync(environment.gradient,environment.tensor->getElementType()); assert(done);
    //Initialize the gradient tensor to zero:
    done = initTensorSync(environment.gradient->getName(),0.0); assert(done);
    //Evaluate the gradient tensor expansion:
    done = evaluateSync(environment.gradient_expansion,environment.gradient); assert(done);
    //Compute the MaxAbs of the gradient tensor:
    double grad_maxabs = 0.0;
    done = computeMaxAbsSync(environment.gradient->getName(),grad_maxabs); assert(done);
    if(grad_maxabs > max_grad_maxabs) max_grad_maxabs = grad_maxabs;
    //Update the optimizable tensor using the computed gradient:
    if(grad_maxabs > tolerance_){
     std::string add_pattern;
     done = generate_addition_pattern(environment.tensor->getRank(),add_pattern,true,
                                      environment.tensor->getName(),environment.gradient->getName()); assert(done);
     done = addTensorsSync(add_pattern,-epsilon_); assert(done);
    }
    //Destroy the gradient tensor:
    done = destroyTensorSync(environment.gradient->getName()); assert(done);
   }
   converged = (max_grad_maxabs <= tolerance_);
  }
  //Compute approximation fidelity:
  done = initTensorSync("_scalar_norm",0.0); assert(done);
  done = evaluateSync(overlap,scalar_norm); assert(done);
  done = computeNorm2Sync("_scalar_norm",fidelity_); assert(done);
  done = destroyTensorSync("_scalar_norm"); assert(done);
 }

 *fidelity = fidelity_;
 return true;
}

} //namespace exatn
