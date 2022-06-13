/** ExaTN::Numerics: Tensor Functor: Initialization of an isometric tensor to unity
REVISION: 2022/06/13

Copyright (C) 2018-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle)
Copyright (C) 2022-2022 NVIDIA Corp. **/

/** Rationale:
 (A) This tensor functor (method) is used to initialize a tensor
     with an isometric group of dimensions to unity. This means
     that the corresponding orthogonal matrix columns are unit vectors,
     where each column represents a flattened space spanned by
     the direct product of the isometric tensor dimensions.
**/

#ifndef EXATN_NUMERICS_FUNCTOR_INIT_UNITY_HPP_
#define EXATN_NUMERICS_FUNCTOR_INIT_UNITY_HPP_

#include "Identifiable.hpp"

#include "tensor_basic.hpp"

#include "tensor_method.hpp" //from TAL-SH

#include <string>
#include <complex>
#include <vector>

#include "errors.hpp"

namespace exatn{

namespace numerics{

class FunctorInitUnity: public talsh::TensorFunctor<Identifiable>{
public:

 FunctorInitUnity(const std::vector<unsigned int> & iso_dims):
  iso_dims_(iso_dims)
 {
 }

 virtual ~FunctorInitUnity() = default;

 virtual const std::string name() const override
 {
  return "TensorFunctorInitUnity";
 }

 virtual const std::string description() const override
 {
  return "Initializes an isometric tensor to unity";
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

 std::vector<unsigned int> iso_dims_; //isometric tensor dimensions

};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_FUNCTOR_INIT_UNITY_HPP_
