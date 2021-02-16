/** ExaTN::Numerics: Tensor Functor: Initialization of Ordering Projection tensors
REVISION: 2021/02/16

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (A) This tensor functor (method) is used to initialize an ordering projection tensor.
     An ordering projection tensor is an even-order tensor with the following elements:
     - Element = 1 if: The first half and the second half of the indices are the same and
                       both are in a monotonically increasing order;
     - Elelent = 0 otherwise.
**/

#ifndef EXATN_NUMERICS_FUNCTOR_INIT_PROJ_HPP_
#define EXATN_NUMERICS_FUNCTOR_INIT_PROJ_HPP_

#include "Identifiable.hpp"

#include "tensor_basic.hpp"

#include "tensor_method.hpp" //from TAL-SH

#include <string>
#include <complex>

#include "errors.hpp"

namespace exatn{

namespace numerics{

class FunctorInitProj: public talsh::TensorFunctor<Identifiable>{
public:

 FunctorInitProj() = default;

 virtual ~FunctorInitProj() = default;

 virtual const std::string name() const override
 {
  return "TensorFunctorInitProj";
 }

 virtual const std::string description() const override
 {
  return "Initializes an Ordering Projection tensor";
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

#endif //EXATN_NUMERICS_FUNCTOR_INIT_PROJ_HPP_
