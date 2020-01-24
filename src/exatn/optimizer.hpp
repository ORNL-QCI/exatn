/** ExaTN:: Optimizer of a closed tensor network expansion functional
REVISION: 2020/01/24

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (A) Given a closed tensor network expansion functional, the tensor network
     expansion optimizer optimizes tensor factors to achieve an extremum of
     that functional.
**/

#ifndef EXATN_OPTIMIZER_HPP_
#define EXATN_OPTIMIZER_HPP_

#include "exatn_numerics.hpp"

#include <memory>

namespace exatn{

class TensorNetworkOptimizer{

public:

 TensorNetworkOptimizer(std::shared_ptr<TensorExpansion> expansion, //inout: closed tensor network expansion functional to optimize
                        double tolerance);                          //in: desired numerical convergence tolerance

 TensorNetworkOptimizer(const TensorNetworkOptimizer &) = default;
 TensorNetworkOptimizer & operator=(const TensorNetworkOptimizer &) = default;
 TensorNetworkOptimizer(TensorNetworkOptimizer &&) noexcept = default;
 TensorNetworkOptimizer & operator=(TensorNetworkOptimizer &&) noexcept = default;
 ~TensorNetworkOptimizer() = default;

 /** Optimizes the given closed tensor network expansion functional.
     Upon success, returns the achieved accuracy of the optimization. **/
 bool optimize(double * accuracy);

 /** Returns the optimized tensor network expansion functional. **/
 std::shared_ptr<TensorExpansion> getSolution(double * accuracy = nullptr);

private:

 std::shared_ptr<TensorExpansion> expansion_; //closed tensor network expansion functional to optimize
 double tolerance_;                           //numerical optimization convergence tolerance
 double accuracy_;                            //actually achieved optimization accuracy
};

} //namespace exatn

#endif //EXATN_OPTIMIZER_HPP_
