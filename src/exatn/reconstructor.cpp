/** ExaTN:: Reconstructor of an approximate tensor network expansion from a given tensor network expansion
REVISION: 2020/03/13

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "reconstructor.hpp"

#include <vector>
#include <cassert>

namespace exatn{

TensorNetworkReconstructor::TensorNetworkReconstructor(std::shared_ptr<TensorExpansion> expansion,
                                                       std::shared_ptr<TensorExpansion> approximant,
                                                       double tolerance):
 expansion_(expansion), approximant_(approximant), tolerance_(tolerance), fidelity_(0.0)
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
 TensorExpansion lagrangian(*approximant_,*expansion_);
 TensorExpansion approximant_conjugate = approximant_->clone(); //deep copy
 approximant_conjugate.conjugate();
 TensorExpansion normalization(*approximant_,approximant_conjugate);
 lagrangian.appendExpansion(normalization,{1.0,0.0});
 //Enumerate optimizable tensors inside the approximant tensor expansion:
 std::vector<TensorExpansion> gradients;
 std::vector<std::pair<std::shared_ptr<Tensor>,TensorElementType>> accumulators;
 // Loop over the tensor networks constituting the approximant tensor expansion:
 for(auto network = approximant_->begin(); network != approximant_->end(); ++network){
  // Loop over the optimizable tensors inside the current tensor network:
  for(auto tensor_conn = network->network_->begin(); tensor_conn != network->network_->end(); ++tensor_conn){
   auto tensor = tensor_conn->second;
   if(tensor.isOptimizable()){ //gradient w.r.t. an optimizable tensor inside the approximant tensor expansion
    gradients.emplace_back(TensorExpansion(*approximant_,tensor.getName(),true));
    accumulators.emplace_back(std::make_pair(std::make_shared<Tensor>("_a"+tensor.getName(),
                                                                      tensor.getShape(),
                                                                      tensor.getSignature()),
                                             tensor.getElementType()));
   }
  }
 }
 //Alternating least squares optimization:
 bool converged = false;
 while(!converged){
  for(unsigned int i = 0; i < gradients.size(); ++i){
   bool created = createTensorSync(accumulators[i].first,accumulators[i].second); assert(created);

  }
 }
 *fidelity = fidelity_;
 return true;
}

} //namespace exatn
