/** ExaTN:: Reconstructor of a tensor network from a tensor expansion
REVISION: 2019/12/14

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "reconstructor.hpp"

namespace exatn{

TensorNetworkReconstructor::TensorNetworkReconstructor(std::shared_ptr<TensorExpansion> expansion,
                                                       std::shared_ptr<TensorExpansion> approximant,
                                                       double tolerance):
 expansion_(expansion), approximant_(approximant), tolerance_(tolerance), fidelity_(0.0)
{
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
 //`Finish
 *fidelity = fidelity_;
 return true;
}

} //namespace exatn
