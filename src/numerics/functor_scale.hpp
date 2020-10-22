/** ExaTN::Numerics: Tensor Functor: Scaling a tensor by a scalar
REVISION: 2019/11/21

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (A) This tensor functor (method) is used to scale a tensor by a scalar.
**/

#ifndef EXATN_NUMERICS_FUNCTOR_SCALE_HPP_
#define EXATN_NUMERICS_FUNCTOR_SCALE_HPP_

#include "Identifiable.hpp"

#include "tensor_basic.hpp"

#include "tensor_method.hpp" //from TAL-SH

#include <complex>
#include <string>

#include "errors.hpp"

namespace exatn{

namespace numerics{

class FunctorScale: public talsh::TensorFunctor<Identifiable>{
public:

 template<typename NumericType>
 FunctorScale(NumericType value): scale_val_(value){}

 virtual ~FunctorScale() = default;

 virtual const std::string name() const override
 {
  return "TensorFunctorScale";
 }

 virtual const std::string description() const override
 {
  return "Scales a tensor by a scalar";
 }

 /** Packs data members into a byte packet. **/
 virtual void pack(BytePacket & packet) override
 {
  appendToBytePacket(&packet,scale_val_.real());
  appendToBytePacket(&packet,scale_val_.imag());
  return;
 }

 /** Unpacks data members from a byte packet. **/
 virtual void unpack(BytePacket & packet) override
 {
  double real,imag;
  extractFromBytePacket(&packet,real);
  extractFromBytePacket(&packet,imag);
  scale_val_ = std::complex<double>{real,imag};
  return;
 }

 /** Scales the local tensor slice by a scalar value.
     Returns zero on success, or an error code otherwise.
     The talsh::Tensor slice is identified by its signature and
     shape that both can be accessed by talsh::Tensor methods. **/
 virtual int apply(talsh::Tensor & local_tensor) override;

private:

 std::complex<double> scale_val_; //scalar scaling value
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_FUNCTOR_SCALE_HPP_
