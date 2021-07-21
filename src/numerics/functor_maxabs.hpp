/** ExaTN::Numerics: Tensor Functor: Computes max-abs norm of a tensor
REVISION: 2021/07/21

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (A) This tensor functor (method) is used to compute the max-abs norm of a tensor,
     that is, the value of the largest-by-absolute-value element of the tensor.
**/

#ifndef EXATN_NUMERICS_FUNCTOR_MAXABS_HPP_
#define EXATN_NUMERICS_FUNCTOR_MAXABS_HPP_

#include "Identifiable.hpp"

#include "tensor_basic.hpp"

#include "tensor_method.hpp" //from TAL-SH

#include <string>
#include <complex>
#include <mutex>

#include "errors.hpp"

namespace exatn{

namespace numerics{

class FunctorMaxAbs: public talsh::TensorFunctor<Identifiable>{
public:

 FunctorMaxAbs(): norm_(0.0) {}

 virtual ~FunctorMaxAbs() = default;

 virtual const std::string name() const override
 {
  return "TensorFunctorMaxAbs";
 }

 virtual const std::string description() const override
 {
  return "Computes max-abs norm of a tensor";
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

 /** Computes max-abs norm of a tensor. Returns zero on success,
     or an error code otherwise. The talsh::Tensor slice is
     identified by its signature and shape that both can be
     accessed by talsh::Tensor methods. **/
 virtual int apply(talsh::Tensor & local_tensor) override;

 /** Returns the tensor norm. **/
 double getNorm() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return norm_;
 }

private:

 double norm_; //computed norm
 static std::mutex mutex_;
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_FUNCTOR_MAXABS_HPP_
