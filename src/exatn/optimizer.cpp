/** ExaTN:: Optimizer of a closed tensor expansion functional
REVISION: 2019/12/14

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "optimizer.hpp"

#include <memory>

namespace exatn{

TensorNetworkOptimizer::TensorNetworkOptimizer(std::shared_ptr<TensorExpansion> expansion,
                                               double tolerance):
 expansion_(expansion), tolerance_(tolerance), accuracy_(-1.0)
{
}


std::shared_ptr<TensorExpansion> TensorNetworkOptimizer::getSolution(double * accuracy)
{
 if(accuracy_ < 0.0) return std::shared_ptr<TensorExpansion>(nullptr);
 if(accuracy != nullptr) *accuracy = accuracy_;
 return expansion_;
}


bool TensorNetworkOptimizer::optimize(double * accuracy)
{
 assert(accuracy != nullptr);
 //`Finish
 *accuracy = accuracy_;
 return true;
}

} //namespace exatn
