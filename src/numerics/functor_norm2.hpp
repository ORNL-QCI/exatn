/** ExaTN::Numerics: Tensor Functor: Computes 2-norm of a tensor
REVISION: 2021/07/21

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (A) This tensor functor (method) is used to compute 2-norm of a tensor.
**/

#ifndef EXATN_NUMERICS_FUNCTOR_NORM2_HPP_
#define EXATN_NUMERICS_FUNCTOR_NORM2_HPP_

#include "Identifiable.hpp"

#include "tensor_basic.hpp"

#include "tensor_method.hpp" //from TAL-SH

#include <string>
#include <complex>
#include <cmath>
#include <mutex>

#include "errors.hpp"

namespace exatn{

namespace numerics{

class FunctorNorm2: public talsh::TensorFunctor<Identifiable>{
public:

 FunctorNorm2(): norm_(0.0) {}

 virtual ~FunctorNorm2() = default;

 virtual const std::string name() const override
 {
  return "TensorFunctorNorm2";
 }

 virtual const std::string description() const override
 {
  return "Computes 2-norm of a tensor";
 }

 /** Packs data members into a byte packet. **/
 virtual void pack(BytePacket & packet) override
 {
  const std::lock_guard<std::mutex> lock(mutex_);
  appendToBytePacket(&packet,norm_);
  return;
 }

 /** Unpacks data members from a byte packet. **/
 virtual void unpack(BytePacket & packet) override
 {
  const std::lock_guard<std::mutex> lock(mutex_);
  extractFromBytePacket(&packet,norm_);
  return;
 }

 /** Computes 2-norm of a tensor. Returns zero on success,
     or an error code otherwise. The talsh::Tensor slice is
     identified by its signature and shape that both can be
     accessed by talsh::Tensor methods. **/
 virtual int apply(talsh::Tensor & local_tensor) override;

 /** Returns the tensor norm. **/
 double getNorm() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return std::sqrt(norm_);
 }

private:

 double norm_; //computed norm
 static std::mutex mutex_;
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_FUNCTOR_NORM2_HPP_
