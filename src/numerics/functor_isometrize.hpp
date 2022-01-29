/** ExaTN::Numerics: Tensor Functor: Tensor Isometrization
REVISION: 2022/01/28

Copyright (C) 2018-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (A) This tensor functor (method) performs the modified Gram-Schmidt
     procedure on a tensor in order to enforce its isometries.
**/

#ifndef EXATN_NUMERICS_FUNCTOR_ISOMETRIZE_HPP_
#define EXATN_NUMERICS_FUNCTOR_ISOMETRIZE_HPP_

#include "Identifiable.hpp"

#include "tensor_basic.hpp"

#include "tensor_method.hpp" //from TAL-SH

#include <string>
#include <vector>
#include <complex>

#include "errors.hpp"

namespace exatn{

namespace numerics{

class FunctorIsometrize: public talsh::TensorFunctor<Identifiable>{
public:

 FunctorIsometrize(const std::vector<unsigned int> & isometry1):
  isometry1_(isometry1)
 {
  for(int i = 1; i < isometry1_.size(); ++i) assert(isometry1_[i] > isometry1_[i-1]);
 }

 FunctorIsometrize(const std::vector<unsigned int> & isometry1,
                   const std::vector<unsigned int> & isometry2):
  isometry1_(isometry1), isometry2_(isometry2)
 {
  for(int i = 1; i < isometry1_.size(); ++i) assert(isometry1_[i] > isometry1_[i-1]);
  for(int i = 1; i < isometry2_.size(); ++i) assert(isometry2_[i] > isometry2_[i-1]);
 }

 virtual ~FunctorIsometrize() = default;

 virtual const std::string name() const override
 {
  return "TensorFunctorIsometrize";
 }

 virtual const std::string description() const override
 {
  return "Enforces tensor isometries";
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

 std::vector<unsigned int> isometry1_;
 std::vector<unsigned int> isometry2_;
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_FUNCTOR_ISOMETRIZE_HPP_
