/** ExaTN::Numerics: Tensor Functor: Computes 1-norm of a tensor
REVISION: 2020/05/02

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (A) This tensor functor (method) is used to compute 1-norm of a tensor.
**/

#ifndef EXATN_NUMERICS_FUNCTOR_NORM1_HPP_
#define EXATN_NUMERICS_FUNCTOR_NORM1_HPP_

#include "Identifiable.hpp"

#include "tensor_basic.hpp"

#include "tensor_method.hpp" //from TAL-SH

#include <string>
#include <complex>

namespace exatn{

namespace numerics{

class FunctorNorm1: public talsh::TensorFunctor<Identifiable>{
public:

 FunctorNorm1(): norm_(0.0) {}

 virtual ~FunctorNorm1() = default;

 virtual const std::string name() const override
 {
  return "TensorFunctorNorm1";
 }

 virtual const std::string description() const override
 {
  return "Computes 1-norm of a tensor";
 }

 /** Packs data members into a byte packet. **/
 virtual void pack(BytePacket & packet) override
 {
  appendToBytePacket(&packet,norm_);
  return;
 }

 /** Unpacks data members from a byte packet. **/
 virtual void unpack(BytePacket & packet) override
 {
  extractFromBytePacket(&packet,norm_);
  return;
 }

 /** Computes 1-norm of a tensor. Returns zero on success,
     or an error code otherwise. The talsh::Tensor slice is
     identified by its signature and shape that both can be
     accessed by talsh::Tensor methods. **/
 virtual int apply(talsh::Tensor & local_tensor) override;

 /** Returns the tensor norm. **/
 double getNorm() const {return norm_;}

private:

 double norm_; //computed norm
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_FUNCTOR_NORM1_HPP_
