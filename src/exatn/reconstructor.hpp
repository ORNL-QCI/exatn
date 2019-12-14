/** ExaTN:: Reconstructor of an approximate tensor expansion from a given tensor expansion
REVISION: 2019/12/14

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:

**/

#ifndef EXATN_RECONSTRUCTOR_HPP_
#define EXATN_RECONSTRUCTOR_HPP_

#include "exatn_numerics.hpp"

#include <memory>

namespace exatn{

class TensorNetworkReconstructor{

public:

 TensorNetworkReconstructor(std::shared_ptr<TensorExpansion> expansion,
                            std::shared_ptr<TensorExpansion> approximant,
                            double tolerance);

 TensorNetworkReconstructor(const TensorNetworkReconstructor &) = default;
 TensorNetworkReconstructor & operator=(const TensorNetworkReconstructor &) = default;
 TensorNetworkReconstructor(TensorNetworkReconstructor &&) noexcept = default;
 TensorNetworkReconstructor & operator=(TensorNetworkReconstructor &&) noexcept = default;
 ~TensorNetworkReconstructor() = default;

 /** Reconstructs a tensor expansion via another tensor expansion approximately.
     Upon success, returns the achieved fidelity of the reconstruction. **/
 bool reconstruct(double * fidelity);

 /** Returns the reconstructing tensor expansion. **/
 std::shared_ptr<TensorExpansion> getSolution(double * fidelity = nullptr);

private:

 std::shared_ptr<TensorExpansion> expansion_;   //tensor expansion to reconstruct
 std::shared_ptr<TensorExpansion> approximant_; //reconstructing tensor expansion
 double tolerance_;                             //reconstruction tolerance
 double fidelity_;                              //actually achieved reconstruction fidelity
};

} //namespace exatn

#endif //EXATN_RECONSTRUCTOR_HPP_
