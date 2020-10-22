/** ExaTN:: Reconstructor of an approximate tensor network expansion from a given tensor network expansion
REVISION: 2020/03/17

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (A) Given a tensor network expansion of some form, the tensor network reconstructor
     optimizes its tensor factors to maximize the overlap with another given constant
     tensor network expansion, thus providing an approximation to it.
     The reconstruction fidelity is the overlap between the two tensor network expansions.
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

 TensorNetworkReconstructor(std::shared_ptr<TensorExpansion> expansion,   //in: tensor expansion to be reconstructed (constant)
                            std::shared_ptr<TensorExpansion> approximant, //inout: reconstructing tensor expansion (unoptimized)
                            double tolerance);                            //in: desired reconstruction convergence tolerance

 TensorNetworkReconstructor(const TensorNetworkReconstructor &) = default;
 TensorNetworkReconstructor & operator=(const TensorNetworkReconstructor &) = default;
 TensorNetworkReconstructor(TensorNetworkReconstructor &&) noexcept = default;
 TensorNetworkReconstructor & operator=(TensorNetworkReconstructor &&) noexcept = default;
 ~TensorNetworkReconstructor() = default;

 /** Approximately reconstructs a tensor network expansion via another tensor network
     expansion. Upon success, returns the achieved fidelity of the reconstruction,
     that is, the overlap between the two tensor network expansions, [0..1]. **/
 bool reconstruct(double * fidelity);

 /** Returns the reconstructing (optimized) tensor network expansion. **/
 std::shared_ptr<TensorExpansion> getSolution(double * fidelity = nullptr);

private:

 struct Environment{
  std::shared_ptr<Tensor> tensor;     //tensor being optimized
  std::shared_ptr<Tensor> gradient;   //gradient w.r.t. the tensor
  TensorExpansion gradient_expansion; //gradient tensor network expansion
 };

 std::shared_ptr<TensorExpansion> expansion_;   //tensor expansion to reconstruct
 std::shared_ptr<TensorExpansion> approximant_; //reconstructing tensor expansion
 double epsilon_;                               //epsilon value for gradient descent
 double tolerance_;                             //numerical reconstruction convergence tolerance
 double fidelity_;                              //actually achieved reconstruction fidelity
 std::vector<Environment> environments_;        //optimization environments for each optimizable tensor
};

} //namespace exatn

#endif //EXATN_RECONSTRUCTOR_HPP_
