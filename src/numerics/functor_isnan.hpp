/** ExaTN::Numerics: Tensor Functor: Checking the tensor on the presence of NaN
REVISION: 2022/09/02

Copyright (C) 2018-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle)
Copyright (C) 2022-2022 NVIDIA Corporation

SPDX-License-Identifier: BSD-3-Clause **/

/** Rationale:
 (A) This tensor functor (method) is used to check the tensor on the presence of NaN
**/

#ifndef EXATN_NUMERICS_FUNCTOR_ISNAN_HPP_
#define EXATN_NUMERICS_FUNCTOR_ISNAN_HPP_

#include "Identifiable.hpp"

#include "tensor_basic.hpp"

#include "tensor_method.hpp" //from TAL-SH

#include <string>
#include <complex>
#include <mutex>

#include "errors.hpp"

namespace exatn{

namespace numerics{


inline bool isnan(float fnum){return std::isnan(fnum);}

inline bool isnan(double fnum){return std::isnan(fnum);}

inline bool isnan(std::complex<float> fnum)
{
 return (std::isnan(fnum.real()) || std::isnan(fnum.imag()));
}

inline bool isnan(std::complex<double> fnum)
{
 return (std::isnan(fnum.real()) || std::isnan(fnum.imag()));
}


class FunctorIsNaN: public talsh::TensorFunctor<Identifiable>{
public:

 FunctorIsNaN(): num_nans_(0)
 {
 }

 virtual ~FunctorIsNaN() = default;

 virtual const std::string name() const override
 {
  return "TensorFunctorIsNaN";
 }

 virtual const std::string description() const override
 {
  return "Checks the tensor on the presence of NaN";
 }

 /** Packs data members into a byte packet. **/
 virtual void pack(BytePacket & packet) override
 {
  const std::lock_guard<std::mutex> lock(mutex_);
  appendToBytePacket(&packet,num_nans_);
  return;
 }

 /** Unpacks data members from a byte packet. **/
 virtual void unpack(BytePacket & packet) override
 {
  const std::lock_guard<std::mutex> lock(mutex_);
  extractFromBytePacket(&packet,num_nans_);
  return;
 }

 /** Computes max-abs norm of a tensor. Returns zero on success,
     or an error code otherwise. The talsh::Tensor slice is
     identified by its signature and shape that both can be
     accessed by talsh::Tensor methods. **/
 virtual int apply(talsh::Tensor & local_tensor) override;

 /** Returns the result of the check **/
 bool nanFree() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return (num_nans_ == 0);
 }

private:

 std::size_t num_nans_; // number of NaNs detected
 static std::mutex mutex_;
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_FUNCTOR_ISNAN_HPP_
