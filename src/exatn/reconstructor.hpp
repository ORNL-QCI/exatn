/** ExaTN:: Reconstructs an approximate tensor network expansion for a given tensor network expansion
REVISION: 2021/01/15

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (A) Given a tensor network expansion of some form, the tensor network reconstructor
     optimizes its tensor factors to maximize the overlap with another given constant
     tensor network expansion, thus providing an approximation to it. The reconstruction
     fidelity is the squared overlap between the two tensor network expansions.
     The reconstruction tolerance is a numerical tolerance used for checking convergence
     of the underlying linear algebra procedures.
 (B) The reconstructed tensor network expansion must be a Ket (primary space) and
     the reconstructing tensor network expansion must be a Bra (dual space).
**/

#ifndef EXATN_RECONSTRUCTOR_HPP_
#define EXATN_RECONSTRUCTOR_HPP_

#include "exatn_numerics.hpp"

#include <memory>
#include <vector>

#include "errors.hpp"

namespace exatn{

class TensorNetworkReconstructor{

public:

 static unsigned int debug;

 static constexpr const double DEFAULT_TOLERANCE = 1e-5;
 static constexpr const double DEFAULT_LEARN_RATE = 0.5;
 static constexpr const unsigned int DEFAULT_MAX_ITERATIONS = 1000;

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
 bool reconstruct(double * residual_norm,  //out: 2-norm of the residual tensor (error)
                  double * fidelity,       //out: squared overlap
                  bool nesterov = false);  //in: Nesterov acceleration
 bool reconstruct(const ProcessGroup & process_group, //in: executing process group
                  double * residual_norm,             //out: 2-norm of the residual tensor (error)
                  double * fidelity,                  //out: squared overlap
                  bool nesterov = false);             //in: Nesterov acceleration

 /** Returns the reconstructing (optimized) tensor network expansion. **/
 std::shared_ptr<TensorExpansion> getSolution(double * residual_norm, //out: 2-norm of the residual tensor (error)
                                              double * fidelity);     //out: squared overlap

 static void resetDebugLevel(unsigned int level = 0);

private:

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

 double input_norm_;                            //2-norm of the input tensor expansion
 double output_norm_;                           //2-norm of the approximant tensor expansion
 double residual_norm_;                         //2-norm of the residual tensor after optimization (error)
 double fidelity_;                              //achieved reconstruction fidelity (squared overlap)
 std::vector<Environment> environments_;        //optimization environments for each optimizable tensor
};

} //namespace exatn

#endif //EXATN_RECONSTRUCTOR_HPP_
