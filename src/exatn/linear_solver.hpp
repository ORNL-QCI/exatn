/** ExaTN:: Linear solver over tensor network manifolds
REVISION: 2021/10/21

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (A) Solves a linear system A * x = b, where A is a tensor network operator,
     b is a given tensor network expansion, and x is the unknown tensor
     network expansion sought for.
**/

#ifndef EXATN_LINEAR_SOLVER_HPP_
#define EXATN_LINEAR_SOLVER_HPP_

#include "exatn_numerics.hpp"

#include <vector>
#include <complex>

#include "errors.hpp"

namespace exatn{

class TensorNetworkLinearSolver{

public:

 static unsigned int debug;
 static int focus;

 static constexpr const double DEFAULT_TOLERANCE = 1e-4;
 static constexpr const unsigned int DEFAULT_MAX_ITERATIONS = 1000;

 TensorNetworkLinearSolver(std::shared_ptr<TensorOperator> tensor_operator,   //in: tensor network operator
                           std::shared_ptr<TensorExpansion> rhs_expansion,    //in: right-hand-side tensor network expansion
                           std::shared_ptr<TensorExpansion> vector_expansion, //inout: tensor network expansion forming the bra/ket vectors
                           double tolerance = DEFAULT_TOLERANCE);             //in: desired numerical convergence tolerance

 TensorNetworkLinearSolver(const TensorNetworkLinearSolver &) = default;
 TensorNetworkLinearSolver & operator=(const TensorNetworkLinearSolver &) = default;
 TensorNetworkLinearSolver(TensorNetworkLinearSolver &&) noexcept = default;
 TensorNetworkLinearSolver & operator=(TensorNetworkLinearSolver &&) noexcept = default;
 ~TensorNetworkLinearSolver() = default;

 /** Resets the numerical tolerance. **/
 void resetTolerance(double tolerance = DEFAULT_TOLERANCE);

 /** Resets the max number of macro-iterations. **/
 void resetMaxIterations(unsigned int max_iterations = DEFAULT_MAX_ITERATIONS);

 /** Solves the linear system over tensor network manifolds. **/
 bool solve();
 bool solve(const ProcessGroup & process_group); //in: executing process group

 /** Returns the found tensor network expansion. **/
 std::shared_ptr<TensorExpansion> getSolution() const;

 /** Enables/disables coarse-grain parallelization over tensor networks. **/
 void enableParallelization(bool parallel = true);

 static void resetDebugLevel(unsigned int level = 0,  //in: debug level
                             int focus_process = -1); //in: process to focus on (-1: all)

private:

 std::shared_ptr<TensorOperator> tensor_operator_;   //tensor network operator
 std::shared_ptr<TensorExpansion> rhs_expansion_;    //right-hand-side tensor network expansion
 std::shared_ptr<TensorExpansion> vector_expansion_; //unknown tensor network expansion sought for
 unsigned int max_iterations_;                       //max number of macro-iterations
 double tolerance_;                                  //numerical convergence tolerance (for the gradient)
 bool parallel_;                                     //enables/disables coarse-grain parallelization over tensor networks

 std::shared_ptr<TensorExpansion> opvec_expansion_;  //A * x tensor network expansion
};

} //namespace exatn

#endif //EXATN_LINEAR_SOLVER_HPP_
