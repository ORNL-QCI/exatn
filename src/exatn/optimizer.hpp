/** ExaTN:: Variational optimizer of a closed symmetric tensor network expansion functional
REVISION: 2021/01/21

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

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

#include <vector>
#include <complex>

#include "errors.hpp"

namespace exatn{

class TensorNetworkOptimizer{

public:

 static unsigned int debug;

 static constexpr const double DEFAULT_TOLERANCE = 1e-5;
 static constexpr const double DEFAULT_LEARN_RATE = 0.5;
 static constexpr const unsigned int DEFAULT_MAX_ITERATIONS = 1000;

 TensorNetworkOptimizer(std::shared_ptr<TensorOperator> tensor_operator,   //in: hermitian tensor network operator
                        std::shared_ptr<TensorExpansion> vector_expansion, //inout: tensor network expansion forming the bra/ket vectors
                        double tolerance = DEFAULT_TOLERANCE)     ;        //in: desired numerical convergence tolerance

 TensorNetworkOptimizer(const TensorNetworkOptimizer &) = default;
 TensorNetworkOptimizer & operator=(const TensorNetworkOptimizer &) = default;
 TensorNetworkOptimizer(TensorNetworkOptimizer &&) noexcept = default;
 TensorNetworkOptimizer & operator=(TensorNetworkOptimizer &&) noexcept = default;
 ~TensorNetworkOptimizer() = default;

 /** Resets the numerical tolerance. **/
 void resetTolerance(double tolerance = DEFAULT_TOLERANCE);

 /** Resets the learning rate. **/
 void resetLearningRate(double learn_rate = DEFAULT_LEARN_RATE);

 /** Resets the max number of macro-iterations. **/
 void resetMaxIterations(unsigned int max_iterations = DEFAULT_MAX_ITERATIONS);

 /** Optimizes the given closed symmetric tensor network expansion functional. **/
 bool optimize();
 bool optimize(const ProcessGroup & process_group); //in: executing process group

 /** Returns the optimized tensor network expansion forming the optimal bra/ket vectors. **/
 std::shared_ptr<TensorExpansion> getSolution() const;

 static void resetDebugLevel(unsigned int level = 0);

private:

 struct Environment{
  std::shared_ptr<Tensor> tensor;       //tensor being optimized: x
  std::shared_ptr<Tensor> gradient;     //gradient w.r.t. the tensor being optimized: g
  std::shared_ptr<Tensor> gradient_aux; //partial gradient tensor (intermediate)
  TensorExpansion gradient_expansion;   //gradient tensor network expansion: H|x> - E*S|x> = g
  TensorExpansion operator_gradient;    //operator gradient tensor network expansion: H|x>
  TensorExpansion metrics_gradient;     //metrics gradient tensor network expansion: S|x>
  TensorExpansion hessian_expansion;    //hessian-gradient tensor network expansion: <g|H|g> - E*<g|S|g>
  std::complex<double> gradient_metric_coef_;
  std::complex<double> hessian_metric_coef_;
 };

 std::shared_ptr<TensorOperator> tensor_operator_;   //tensor operator
 std::shared_ptr<TensorExpansion> vector_expansion_; //tensor network expansion to optimize (bra/ket vector)
 unsigned int max_iterations_;                       //max number of macro-iterations
 double epsilon_;                                    //learning rate for the gradient descent based tensor update
 double tolerance_;                                  //numerical convergence tolerance (for the gradient)

 std::vector<Environment> environments_;             //optimization environments for each optimizable tensor
};

} //namespace exatn

#endif //EXATN_OPTIMIZER_HPP_
