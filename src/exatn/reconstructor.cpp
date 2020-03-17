/** ExaTN:: Reconstructor of an approximate tensor network expansion from a given tensor network expansion
REVISION: 2020/03/17

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "reconstructor.hpp"

#include <unordered_set>
#include <string>
#include <cassert>

namespace exatn{

TensorNetworkReconstructor::TensorNetworkReconstructor(std::shared_ptr<TensorExpansion> expansion,
                                                       std::shared_ptr<TensorExpansion> approximant,
                                                       double tolerance):
 expansion_(expansion), approximant_(approximant), epsilon_(0.1), tolerance_(tolerance), fidelity_(0.0)
{
 assert(expansion_->isKet() && approximant_->isBra());
 assert(expansion_->getRank() == approximant_->getRank());
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
 TensorExpansion lagrangian(*approximant_,*expansion_); // <approximant|expansion>
 TensorExpansion approximant_ket(*approximant_,false); // <approximant|
 approximant_ket.conjugate(); // |approximant>
 TensorExpansion normalization(*approximant_,approximant_ket); // <approximant|approximant>
 //lagrangian.appendExpansion(normalization,{1.0,0.0});

 //Prepare optimization environments for all optimizable tensors in the approximant:
 std::unordered_set<std::string> tensor_names;
 // Loop over the tensor networks constituting the approximant tensor expansion:
 for(auto network = approximant_->cbegin(); network != approximant_->cend(); ++network){
  // Loop over the optimizable tensors inside the current tensor network:
  for(auto tensor_conn = network->network_->begin(); tensor_conn != network->network_->end(); ++tensor_conn){
   auto tensor = tensor_conn->second;
   if(tensor.isOptimizable()){ //gradient w.r.t. an optimizable tensor inside the approximant tensor expansion
    auto res = tensor_names.emplace(tensor.getName());
    if(res.second){
     environments_.emplace_back(Environment{tensor.getTensor(),
                                            std::make_shared<Tensor>("_a"+tensor.getName(),
                                                                     tensor.getShape(),
                                                                     tensor.getSignature()),
                                            TensorExpansion(lagrangian,tensor.getName(),true)
                                           });
    }
   }
  }
 }
 //Optimization procedure:
 bool converged = (environments_.size() == 0);
 // Create a scalar tensor:
 if(!converged){
  auto scalar_norm = makeSharedTensor("_scalar_norm");
  bool done = createTensorSync(scalar_norm,environments_[0].tensor->getElementType()); assert(done);
  while(!converged){
   for(auto & environment: environments_){
    //Create the gradient tensor:
    done = createTensorSync(environment.gradient,environment.tensor->getElementType()); assert(done);
    //Initialize the gradient tensor to zero:
    done = initTensorSync(environment.gradient->getName(),0.0); assert(done);
    //Evaluate the gradient tensor expansion:
    done = evaluateSync(environment.gradient_expansion,environment.gradient); assert(done);
    //Update the optimizable tensor using the computed gradient (conjugated):
    std::string add_pattern;
    done = generate_addition_pattern(environment.tensor->getRank(),add_pattern,true,
                                     environment.tensor->getName(),environment.gradient->getName()); assert(done);
    done = addTensors(add_pattern,epsilon_); assert(done);
    //Compute the norm of the approximant:
    done = initTensorSync("_scalar_norm",0.0); assert(done);
    done = evaluateSync(normalization,scalar_norm); assert(done);
    //Re-normalize the optimizable tensor:
    
    //Destroy the gradient tensor:
    done = destroyTensorSync(environment.gradient->getName()); assert(done);
   }
  }
  done = destroyTensorSync("_scalar_norm"); assert(done);
 }
 *fidelity = fidelity_;
 return true;
}

} //namespace exatn
