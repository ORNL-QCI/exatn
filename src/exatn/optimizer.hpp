/** ExaTN:: Optimizer of a closed tensor expansion functional
REVISION: 2019/12/18

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:

**/

#ifndef EXATN_OPTIMIZER_HPP_
#define EXATN_OPTIMIZER_HPP_

#include "exatn_numerics.hpp"

#include <memory>

namespace exatn{

class TensorNetworkOptimizer{

public:

 TensorNetworkOptimizer(std::shared_ptr<TensorExpansion> expansion, //in: closed tensor expansion functional to be optimized
                        double tolerance);                          //in: desired numerical convergence tolerance

 TensorNetworkOptimizer(const TensorNetworkOptimizer &) = default;
 TensorNetworkOptimizer & operator=(const TensorNetworkOptimizer &) = default;
 TensorNetworkOptimizer(TensorNetworkOptimizer &&) noexcept = default;
 TensorNetworkOptimizer & operator=(TensorNetworkOptimizer &&) noexcept = default;
 ~TensorNetworkOptimizer() = default;

 /** Optimizes the given closed tensor expansion functional. Upon success,
     returns the achieved accuracy of the optimization. **/
 bool optimize(double * accuracy);

 /** Returns the optimized tensor expansion. **/
 std::shared_ptr<TensorExpansion> getSolution(double * accuracy = nullptr);

private:

 std::shared_ptr<TensorExpansion> expansion_; //closed tensor expansion functional to optimize
 double tolerance_;                           //optimization convergence tolerance
 double accuracy_;                            //actually achieved optimization accuracy
};

} //namespace exatn

#endif //EXATN_OPTIMIZER_HPP_
