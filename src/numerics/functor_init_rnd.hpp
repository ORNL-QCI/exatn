/** ExaTN::Numerics: Tensor Functor: Initialization to a random value
REVISION: 2019/11/21

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (A) This tensor functor (method) is used to initialize a Tensor to a random value.
**/

#ifndef EXATN_NUMERICS_FUNCTOR_INIT_RND_HPP_
#define EXATN_NUMERICS_FUNCTOR_INIT_RND_HPP_

#include "Identifiable.hpp"

#include "tensor_basic.hpp"

#include "tensor_method.hpp" //from TAL-SH

#include <string>

#include "errors.hpp"

namespace exatn{

namespace numerics{

class FunctorInitRnd: public talsh::TensorFunctor<Identifiable>{
public:

 FunctorInitRnd() = default;

 virtual ~FunctorInitRnd() = default;

 virtual const std::string name() const override
 {
  return "TensorFunctorInitRnd";
 }

 virtual const std::string description() const override
 {
  return "Initializes a tensor to a random value";
 }

 /** Packs data members into a byte packet. **/
 virtual void pack(BytePacket & packet) override
 {
  return;
 }

 /** Unpacks data members from a byte packet. **/
 virtual void unpack(BytePacket & packet) override
 {
  return;
 }

 /** Initializes the local tensor slice to a value.
     Returns zero on success, or an error code otherwise.
     The talsh::Tensor slice is identified by its signature and
     shape that both can be accessed by talsh::Tensor methods. **/
 virtual int apply(talsh::Tensor & local_tensor) override;

};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_FUNCTOR_INIT_RND_HPP_
