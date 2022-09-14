/** ExaTN:: Reconstructs an approximate tensor network expansion for a given tensor network expansion
REVISION: 2022/09/13

Copyright (C) 2018-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle)
Copyright (C) 2022-2022 NVIDIA Corporation

SPDX-License-Identifier: BSD-3-Clause **/

/** Rationale:
 (A) Given a tensor network expansion of some form, the tensor network reconstructor
     optimizes its tensor factors to maximize the overlap with another given constant
     tensor network expansion, thus providing an approximation to it. The reconstruction
     fidelity is the normalized squared overlap between the two tensor network expansions.
     The reconstruction tolerance is a numerical tolerance used for checking convergence
     of the underlying linear algebra procedures.
 (B) The reconstructed tensor network expansion must be a Ket (primary space) and
     the reconstructing tensor network expansion must be a Bra (dual space).
**/

#ifndef EXATN_RECONSTRUCTOR_HPP_
#define EXATN_RECONSTRUCTOR_HPP_

#include "exatn_numerics.hpp"

#include <vector>
#include <complex>

#include "errors.hpp"

namespace exatn{

class TensorNetworkReconstructor{

public:

 static unsigned int debug;
 static int focus;

 static constexpr double DEFAULT_TOLERANCE = 1e-5;
 static constexpr double DEFAULT_LEARN_RATE = 0.5;
 static constexpr unsigned int DEFAULT_MAX_ITERATIONS = 1000;
 static constexpr unsigned int DEFAULT_OVERLAP_ITERATIONS = 3;
 static constexpr double DEFAULT_ACCEPTABLE_FIDELITY = 0.01;
 static constexpr double DEFAULT_MIN_INITIAL_OVERLAP = 1e-9;
 static constexpr double DEFAULT_GRAD_ZERO_THRESHOLD = 1e-7;

 TensorNetworkReconstructor(std::shared_ptr<TensorExpansion> expansion,   //in: tensor network expansion to be reconstructed (constant)
                            std::shared_ptr<TensorExpansion> approximant, //inout: reconstructing tensor network expansion
                            double tolerance = DEFAULT_TOLERANCE);        //in: desired reconstruction convergence tolerance

 TensorNetworkReconstructor(const TensorNetworkReconstructor &) = default;
 TensorNetworkReconstructor & operator=(const TensorNetworkReconstructor &) = default;
 TensorNetworkReconstructor(TensorNetworkReconstructor &&) noexcept = default;
 TensorNetworkReconstructor & operator=(TensorNetworkReconstructor &&) noexcept = default;
 ~TensorNetworkReconstructor() = default;

 /** Resets the reconstruction tolerance. **/
 void resetTolerance(double tolerance = DEFAULT_TOLERANCE);

 /** Resets the learning rate. **/
 void resetLearningRate(double learn_rate = DEFAULT_LEARN_RATE);

 /** Resets the max number of macro-iterations (sweeping epochs). **/
 void resetMaxIterations(unsigned int max_iterations = DEFAULT_MAX_ITERATIONS);

 /** Approximately reconstructs a tensor network expansion via another tensor network
     expansion. Upon success, returns the achieved fidelity of the reconstruction,
     that is, the squared overlap between the two tensor network expansions: [0..1]. **/
 bool reconstruct(double * residual_norm,             //out: 2-norm of the residual tensor (error)
                  double * fidelity,                  //out: squared normalized overlap (fidelity)
                  bool rnd_init = true,               //in: random initialization flag
                  bool nesterov = true,               //in: Nesterov acceleration
                  bool isometric = false,             //in: whether or not to use the fully isometric solver
                  double acceptable_fidelity = DEFAULT_ACCEPTABLE_FIDELITY); //in: acceptable fidelity

 bool reconstruct(const ProcessGroup & process_group, //in: executing process group
                  double * residual_norm,             //out: 2-norm of the residual tensor (error)
                  double * fidelity,                  //out: squared normalized overlap (fidelity)
                  bool rnd_init = true,               //in: random initialization flag
                  bool nesterov = true,               //in: Nesterov acceleration
                  bool isometric = false,             //in: whether or not to use the fully isometric solver
                  double acceptable_fidelity = DEFAULT_ACCEPTABLE_FIDELITY); //in: acceptable fidelity

 /** Returns the reconstructing (optimized) tensor network expansion. **/
 std::shared_ptr<TensorExpansion> getSolution(double * residual_norm,   //out: 2-norm of the residual tensor (error)
                                              double * fidelity) const; //out: squared normalized overlap (fidelity)

 /** Enables/disables coarse-grain parallelization over tensor networks. **/
 void enableParallelization(bool parallel = true);

 static void resetDebugLevel(unsigned int level = 0,  //in: debug level
                             int focus_process = -1); //in: process to focus on (-1: all)

protected:

 //Implementation based on the steepest descent algorithm:
 bool reconstruct_sd(const ProcessGroup & process_group, //in: executing process group
                     double * residual_norm,             //out: 2-norm of the residual tensor (error)
                     double * fidelity,                  //out: squared normalized overlap (fidelity)
                     bool rnd_init,                      //in: random initialization flag
                     bool nesterov,                      //in: Nesterov acceleration
                     double acceptable_fidelity);        //in: acceptable fidelity

 //Implementation based on the steepest descent algorithm for isometric tensor networks:
 bool reconstruct_iso_sd(const ProcessGroup & process_group, //in: executing process group
                         double * residual_norm,             //out: 2-norm of the residual tensor (error)
                         double * fidelity,                  //out: squared normalized overlap (fidelity)
                         bool rnd_init,                      //in: random initialization flag
                         double acceptable_fidelity);        //in: acceptable fidelity

private:

 void reinitializeApproximant(const ProcessGroup & process_group);

 struct Environment{
  std::shared_ptr<Tensor> tensor;       //tensor being optimized
  std::shared_ptr<Tensor> tensor_aux;   //auxiliary tensor (e.g., previous iteration)
  std::shared_ptr<Tensor> gradient;     //gradient w.r.t. the tensor being optimized
  std::shared_ptr<Tensor> gradient_aux; //auxiliary gradient (e.g., previous iteration)
  TensorExpansion gradient_expansion;   //gradient tensor network expansion
  TensorExpansion hessian_expansion;    //hessian-gradient tensor network expansion
 };

 std::shared_ptr<TensorExpansion> expansion_;   //tensor network expansion to reconstruct
 std::shared_ptr<TensorExpansion> approximant_; //reconstructing tensor network expansion
 unsigned int max_iterations_;                  //max number of macro-iterations
 double epsilon_;                               //learning rate for the gradient descent based tensor update
 double tolerance_;                             //numerical reconstruction convergence tolerance (for the gradient)
 bool parallel_;                                //enables/disables coarse-grain parallelization over tensor networks

 double input_norm_;                            //2-norm of the input tensor expansion
 double output_norm_;                           //2-norm of the approximant tensor expansion
 double residual_norm_;                         //2-norm of the residual tensor after optimization (error)
 double fidelity_;                              //achieved reconstruction fidelity (normalized squared overlap)

 std::vector<Environment> environments_;        //optimization environments for each optimizable tensor
};

} //namespace exatn

#endif //EXATN_RECONSTRUCTOR_HPP_
