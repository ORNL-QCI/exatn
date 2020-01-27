/** ExaTN:: Variational optimizer of a closed symmetric tensor network expansion functional
REVISION: 2020/01/26

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (A) Given a closed symmetric tensor network expansion functional, composed
     of some tensor operator closed from both sides by the bra and ket
     vectors formed by the same tensor network expansion, this tensor network
     variational optimizer will optimize the tensor factors constituting the
     bra/ket tensor network vectors to arrive at an extremum of that functional.
**/

#ifndef EXATN_OPTIMIZER_HPP_
#define EXATN_OPTIMIZER_HPP_

#include "exatn_numerics.hpp"

#include <memory>

namespace exatn{

class TensorNetworkOptimizer{

public:

 TensorNetworkOptimizer(std::shared_ptr<TensorOperator> tensor_operator,   //in: hermitian tensor network operator
                        std::shared_ptr<TensorExpansion> vector_expansion, //inout: tensor network expansion forming the bra/ket vectors
                        double tolerance);                                 //in: desired numerical convergence tolerance

 TensorNetworkOptimizer(const TensorNetworkOptimizer &) = default;
 TensorNetworkOptimizer & operator=(const TensorNetworkOptimizer &) = default;
 TensorNetworkOptimizer(TensorNetworkOptimizer &&) noexcept = default;
 TensorNetworkOptimizer & operator=(TensorNetworkOptimizer &&) noexcept = default;
 ~TensorNetworkOptimizer() = default;

 /** Optimizes the given closed symmetric tensor network expansion functional. **/
 bool optimize();

 /** Returns the optimized tensor network expansion forming the optimal bra/ket vectors. **/
 std::shared_ptr<TensorExpansion> getSolution();

private:

 std::shared_ptr<TensorOperator> tensor_operator_;   //tensor operator
 std::shared_ptr<TensorExpansion> vector_expansion_; //tensor network expansion to optimize (bra/ket vector)
 double tolerance_;                                  //desired numerical optimization convergence tolerance
};

} //namespace exatn

#endif //EXATN_OPTIMIZER_HPP_
