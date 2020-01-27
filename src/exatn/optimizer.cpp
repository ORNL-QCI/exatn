/** ExaTN:: Variational optimizer of a closed symmetric tensor network expansion functional
REVISION: 2020/01/26

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "optimizer.hpp"

#include <cassert>

namespace exatn{

TensorNetworkOptimizer::TensorNetworkOptimizer(std::shared_ptr<TensorOperator> tensor_operator,
                                               std::shared_ptr<TensorExpansion> vector_expansion,
                                               double tolerance):
 tensor_operator_(tensor_operator), vector_expansion_(vector_expansion), tolerance_(tolerance)
{
}


std::shared_ptr<TensorExpansion> TensorNetworkOptimizer::getSolution()
{
 return vector_expansion_;
}


bool TensorNetworkOptimizer::optimize()
{
 //`Finish
 return true;
}

} //namespace exatn
