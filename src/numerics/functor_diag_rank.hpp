/** ExaTN::Numerics: Tensor Functor: Computes partial 2-norms over a given tensor dimension
REVISION: 2020/04/23

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (A) Given a tensor D(a,b,c,k) and a specific dimension of it, say k,
     computes partial 2-norms L(k) = D+(a,b,c,k) * D(a,b,c,k), for all k.
     This procedure is useful in re-establishing a low-rank structure
     of tensor D(a,b,c,k) in case it came out as a factor from SVD
     with absorbed singular values.
**/

#ifndef EXATN_NUMERICS_FUNCTOR_DIAG_RANK_HPP_
#define EXATN_NUMERICS_FUNCTOR_DIAG_RANK_HPP_

#include "Identifiable.hpp"

#include "tensor_basic.hpp"
#include "tensor_shape.hpp"

#include "tensor_method.hpp" //from TAL-SH

#include <type_traits>
#include <string>
#include <vector>
#include <complex>

#include <cassert>

namespace exatn{

namespace numerics{

class FunctorDiagRank: public talsh::TensorFunctor<Identifiable>{
public:

 FunctorDiagRank(unsigned int tensor_dimension); //in: chosen tensor dimension

 virtual ~FunctorDiagRank() = default;

 virtual const std::string name() const override
 {
  return "TensorFunctorDiagRank";
 }

 virtual const std::string description() const override
 {
  return "Diagnoses tensor rank over a given dimension";
 }

 /** Packs data members into a byte packet. **/
 virtual void pack(BytePacket & packet) override;

 /** Unpacks data members from a byte packet. **/
 virtual void unpack(BytePacket & packet) override;

 /** Computes partial 2-norms over a given tensor dimension.
     Returns zero on success, or an error code otherwise.
     The talsh::Tensor slice is identified by its signature and
     shape that both can be accessed by talsh::Tensor methods. **/
 virtual int apply(talsh::Tensor & local_tensor) override;

private:

 unsigned int tensor_dimension_;     //specific tensor dimension: [0..order-1]
 std::vector<double> partial_norms_; //partial norms over the chosen tensor dimension
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_FUNCTOR_DIAG_RANK_HPP_
