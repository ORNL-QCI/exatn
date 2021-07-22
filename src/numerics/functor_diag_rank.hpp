/** ExaTN::Numerics: Tensor Functor: Computes partial 2-norms over a given tensor dimension
REVISION: 2021/07/22

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

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
#include <mutex>

#include "errors.hpp"

namespace exatn{

namespace numerics{

class FunctorDiagRank: public talsh::TensorFunctor<Identifiable>{
public:

 FunctorDiagRank(unsigned int tensor_dimension, //in: chosen tensor dimension
                 DimExtent dimension_extent,    //in: tensor dimension extent
                 DimOffset dimension_base = 0); //in: tensor dimension base (for sliced dimensions)

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

 const std::vector<double> & getPartialNorms() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return partial_norms_;
 }

private:

 unsigned int tensor_dimension_;     //specific tensor dimension: [0..order-1]
 DimOffset dimension_base_;          //dimension base offset (if the dimension is sliced)
 std::vector<double> partial_norms_; //partial norms over the chosen tensor dimension
 static std::mutex mutex_;
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_FUNCTOR_DIAG_RANK_HPP_
