/** ExaTN:: Reconstructs an approximate tensor network operator for a given tensor network operator
REVISION: 2021/10/22

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (A) Given a tensor network operator of some form, the tensor network operator remapper
     optimizes its tensor factors to maximize the overlap with another given constant
     tensor network operator, thus providing an approximation to it. The reconstruction
     fidelity is the normalized squared overlap between the two tensor network operators.
     The reconstruction tolerance is a numerical tolerance used for checking convergence
     of the underlying linear algebra procedures.
**/

#ifndef EXATN_REMAPPER_HPP_
#define EXATN_REMAPPER_HPP_

#include "exatn_numerics.hpp"

#include <memory>

#include "errors.hpp"

namespace exatn{

class TensorOperatorRemapper{

public:

 static unsigned int debug;
 static int focus;

 static constexpr const double DEFAULT_TOLERANCE = 1e-5;
 static constexpr const unsigned int DEFAULT_MAX_ITERATIONS = 1000;
 static constexpr const double DEFAULT_ACCEPTABLE_FIDELITY = 0.01;

 TensorOperatorRemapper(std::shared_ptr<Tensor> ket_space,           //in: tensor defining the ket space
                        std::shared_ptr<Tensor> bra_space,           //in: tensor defining the bra space
                        std::shared_ptr<TensorOperator> target,      //in: tensor network operator to be reconstructed (constant)
                        std::shared_ptr<TensorOperator> approximant, //inout: reconstructing tensor network operator
                        double tolerance = DEFAULT_TOLERANCE);       //in: desired reconstruction convergence tolerance

 TensorOperatorRemapper(std::shared_ptr<Tensor> ket_space,           //in: tensor defining the ket space (if symmetric to bra)
                        std::shared_ptr<TensorOperator> target,      //in: tensor network operator to be reconstructed (constant)
                        std::shared_ptr<TensorOperator> approximant, //inout: reconstructing tensor network operator
                        double tolerance = DEFAULT_TOLERANCE);       //in: desired reconstruction convergence tolerance

 TensorOperatorRemapper(const TensorOperatorRemapper &) = default;
 TensorOperatorRemapper & operator=(const TensorOperatorRemapper &) = default;
 TensorOperatorRemapper(TensorOperatorRemapper &&) noexcept = default;
 TensorOperatorRemapper & operator=(TensorOperatorRemapper &&) noexcept = default;
 ~TensorOperatorRemapper() = default;

 /** Resets the reconstruction tolerance. **/
 void resetTolerance(double tolerance = DEFAULT_TOLERANCE);

 /** Resets the max number of macro-iterations (sweeping epochs). **/
 void resetMaxIterations(unsigned int max_iterations = DEFAULT_MAX_ITERATIONS);

 /** Approximately reconstructs a tensor network operator via another tensor network
     operator. Upon success, returns the achieved fidelity of the reconstruction,
     that is, the squared overlap between the two tensor network expansions: [0..1]. **/
 bool reconstruct(double * residual_norm,             //out: 2-norm of the residual tensor (error)
                  double * fidelity,                  //out: squared normalized overlap (fidelity)
                  bool rnd_init = true,               //in: random initialization flag
                  bool nesterov = true,               //in: Nesterov acceleration
                  double acceptable_fidelity = DEFAULT_ACCEPTABLE_FIDELITY); //in: acceptable fidelity

 bool reconstruct(const ProcessGroup & process_group, //in: executing process group
                  double * residual_norm,             //out: 2-norm of the residual tensor (error)
                  double * fidelity,                  //out: squared normalized overlap (fidelity)
                  bool rnd_init = true,               //in: random initialization flag
                  bool nesterov = true,               //in: Nesterov acceleration
                  double acceptable_fidelity = DEFAULT_ACCEPTABLE_FIDELITY); //in: acceptable fidelity

 /** Returns the reconstructing (optimized) tensor network operator. **/
 std::shared_ptr<TensorOperator> getSolution(double * residual_norm,   //out: 2-norm of the residual tensor (error)
                                             double * fidelity) const; //out: squared normalized overlap (fidelity)

 /** Enables/disables coarse-grain parallelization over tensor networks. **/
 void enableParallelization(bool parallel = true);

 static void resetDebugLevel(unsigned int level = 0,  //in: debug level
                             int focus_process = -1); //in: process to focus on (-1: all)

private:

 std::shared_ptr<Tensor> ket_space_;           //tensor defining the ket space
 std::shared_ptr<Tensor> bra_space_;           //tensor defining the bra space
 std::shared_ptr<TensorOperator> target_;      //tensor network operator to reconstruct
 std::shared_ptr<TensorOperator> approximant_; //reconstructing tensor network operator
 unsigned int max_iterations_;                 //max number of macro-iterations
 double tolerance_;                            //numerical reconstruction convergence tolerance (for the gradient)
 bool parallel_;                               //enables/disables coarse-grain parallelization over tensor networks

 double residual_norm_;                        //2-norm of the residual tensor after optimization (error)
 double fidelity_;                             //achieved reconstruction fidelity (normalized squared overlap)
};

} //namespace exatn

#endif //EXATN_REMAPPER_HPP_
