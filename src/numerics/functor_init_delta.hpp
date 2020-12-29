/** ExaTN::Numerics: Tensor Functor: Initialization of Kronecker Delta tensors
REVISION: 2020/12/29

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (A) This tensor functor (method) is used to initialize a Kronecker Delta tensor.
     A Kronecker Delta tensor is defined as an arbitrary order tensor with all
     elements equal to zero except those which refer to multi-indices with all
     indices having the same value, in which case the element value is 1.
**/

#ifndef EXATN_NUMERICS_FUNCTOR_INIT_DELTA_HPP_
#define EXATN_NUMERICS_FUNCTOR_INIT_DELTA_HPP_

#include "Identifiable.hpp"

#include "tensor_basic.hpp"

#include "tensor_method.hpp" //from TAL-SH

#include <string>
#include <complex>

#include "errors.hpp"

namespace exatn{

namespace numerics{

class FunctorInitDelta: public talsh::TensorFunctor<Identifiable>{
public:

 FunctorInitDelta() = default;

 virtual ~FunctorInitDelta() = default;

 virtual const std::string name() const override
 {
  return "TensorFunctorInitDelta";
 }

 virtual const std::string description() const override
 {
  return "Initializes a Kronecker Delta tensor";
 }

 /** Packs data members into a byte packet. **/
 virtual void pack(BytePacket & packet) override;

 /** Unpacks data members from a byte packet. **/
 virtual void unpack(BytePacket & packet) override;

 /** Initializes the local tensor slice with external data.
     Returns zero on success, or an error code otherwise.
     The talsh::Tensor slice is identified by its signature and
     shape that both can be accessed by talsh::Tensor methods. **/
 virtual int apply(talsh::Tensor & local_tensor) override;

private:

};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_FUNCTOR_INIT_DELTA_HPP_
