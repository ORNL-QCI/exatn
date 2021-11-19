/** ExaTN:: Variational optimizer of a closed symmetric tensor network expansion functional
REVISION: 2021/11/19

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (A) Given a closed symmetric tensor network expansion functional, composed
     of some tensor operator closed from both sides by the bra and ket
     vectors formed by the same tensor network expansion, this tensor network
     variational optimizer will optimize the tensor factors constituting the
     bra/ket tensor network vectors to arrive at an extremum of that functional,
     targeting its minimum (or maximum):
      E = <x|H|x> / <x|x>, where H is a tensor network operator, and x is a
      tensor network expansion that delivers an extremum to the functional.
**/

#ifndef EXATN_OPTIMIZER_HPP_
#define EXATN_OPTIMIZER_HPP_

#include "exatn_numerics.hpp"
#include "reconstructor.hpp"

#include <vector>
#include <complex>

#include "errors.hpp"

namespace exatn{

class TensorNetworkOptimizer{

public:

 static unsigned int debug;
 static int focus;

 static constexpr const double DEFAULT_TOLERANCE = 1e-4;
 static constexpr const double DEFAULT_LEARN_RATE = 0.5;
 static constexpr const unsigned int DEFAULT_MAX_ITERATIONS = 1000;
 static constexpr const unsigned int DEFAULT_MICRO_ITERATIONS = 1;
 static constexpr const bool PREOPTIMIZE_INITIAL_GUESS = false;
 static constexpr const unsigned int DEFAULT_KRYLOV_GUESS_DIM = 8;
 static constexpr const unsigned int DEFAULT_GUESS_MAX_BOND_DIM = DEFAULT_KRYLOV_GUESS_DIM;
 static constexpr const double DEFAULT_GUESS_TOLERANCE = 1e-3;

 TensorNetworkOptimizer(std::shared_ptr<TensorOperator> tensor_operator,   //in: hermitian tensor network operator
                        std::shared_ptr<TensorExpansion> vector_expansion, //inout: tensor network expansion forming the bra/ket vectors
                        double tolerance = DEFAULT_TOLERANCE);             //in: desired numerical convergence tolerance

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

 /** Resets the number of micro-iterations. **/
 void resetMicroIterations(unsigned int micro_iterations = DEFAULT_MICRO_ITERATIONS);

 /** Optimizes the given closed symmetric tensor network expansion
     functional for its minimum (or maximum). **/
 bool optimize();

 bool optimize(const ProcessGroup & process_group); //in: executing process group

 /** Returns the optimized tensor network expansion forming the optimal
     bra/ket vectors delivering an extremum to the functional. **/
 std::shared_ptr<TensorExpansion> getSolution(std::complex<double> * average_expect_val = nullptr) const;

 /** Returns the achieved expectation value of the optimized functional. **/
 std::complex<double> getExpectationValue() const;

 /** Performs a consecutive tensor network functional optimization
     delivering multiple extreme eigenvalues/eigenvectors. **/
 bool optimize(unsigned int num_roots);            //in: number of extreme roots to find

 bool optimize(const ProcessGroup & process_group, //in: executing process group
               unsigned int num_roots);            //in: number of extreme roots to find

 /** Returns a specific extreme root (eigenvalue/eigenvector pair). **/
 std::shared_ptr<TensorExpansion> getSolution(unsigned int root_id,
                                              std::complex<double> * average_expect_val = nullptr) const;

 /** Returns a specific extreme eigenvalue. **/
 std::complex<double> getExpectationValue(unsigned int root_id) const;

 /** Enables/disables coarse-grain parallelization over tensor networks. **/
 void enableParallelization(bool parallel = true);

 static void resetDebugLevel(unsigned int level = 0,  //in: debug level
                             int focus_process = -1); //in: process to focus on (-1: all)

protected:

 //Generates a pre-optimized initial guess for the extreme eigen-root (lowest by default):
 void computeInitialGuess(const ProcessGroup & process_group,
                          bool highest = false,
                          unsigned int guess_dim = DEFAULT_KRYLOV_GUESS_DIM);

 //Implementation based on the steepest descent algorithm:
 bool optimize_sd(const ProcessGroup & process_group); //in: executing process group

private:

 struct Environment{
  std::shared_ptr<Tensor> tensor;       //tensor being optimized: x
  std::shared_ptr<Tensor> gradient;     //gradient w.r.t. the tensor being optimized: g
  std::shared_ptr<Tensor> gradient_aux; //partial gradient tensor (intermediate)
  TensorExpansion gradient_expansion;   //gradient tensor network expansion: H|x> - E*S|x> = g
  TensorExpansion operator_gradient;    //operator gradient tensor network expansion: H|x>
  TensorExpansion metrics_gradient;     //metrics gradient tensor network expansion: S|x>
  TensorExpansion hessian_expansion;    //hessian-gradient tensor network expansion: <g|H|g> - E*<g|S|g>
  std::complex<double> expect_value;    //current expectation value
 };

 std::vector<std::shared_ptr<TensorExpansion>> eigenvectors_; //extreme eigenvectors
 std::vector<std::complex<double>> eigenvalues_;              //extreme eigenvalues

 std::shared_ptr<TensorOperator> tensor_operator_;   //tensor network operator
 std::shared_ptr<TensorExpansion> vector_expansion_; //tensor network expansion to optimize (bra/ket vector)
 unsigned int max_iterations_;                       //max number of macro-iterations
 unsigned int micro_iterations_;                     //number of microiterations per optimized tensor
 double epsilon_;                                    //learning rate for the gradient descent based tensor update
 double tolerance_;                                  //numerical convergence tolerance (for the gradient)
 bool parallel_;                                     //enables/disables coarse-grain parallelization over tensor networks

 std::complex<double> average_expect_val_;           //average expectation value (across all optimized tensor factors)

 std::vector<Environment> environments_;             //optimization environments for each optimizable tensor
};

} //namespace exatn

#endif //EXATN_OPTIMIZER_HPP_
