/** ExaTN::Numerics: Tensor Functor: Initialization to a given external data
REVISION: 2019/12/08

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (A) This tensor functor (method) is used to initialize a Tensor
     with some externally provided data. This tensor functor should
     normally be used for initializing small tensors.
 (B) The external data vector used to construct this tensor functor
     must contain the entire tensor body stored column-major, with
     the explicit shape of the full tensor supplied as well.
**/

#ifndef EXATN_NUMERICS_FUNCTOR_INIT_DAT_HPP_
#define EXATN_NUMERICS_FUNCTOR_INIT_DAT_HPP_

#include "Identifiable.hpp"

#include "tensor_basic.hpp"
#include "tensor_shape.hpp"

#include "tensor_method.hpp" //from TAL-SH

#include <type_traits>
#include <string>
#include <vector>
#include <complex>

#include <cassert>

namespace exatn{

namespace numerics{

class FunctorInitDat: public talsh::TensorFunctor<Identifiable>{
public:

 /** TensorShape object must specify the shape of the full tensor and
     the given external data vector must contain the full tensor body
     stored column-major as specified by the provided tensor shape. **/
 template <typename NumericType>
 FunctorInitDat(const TensorShape & full_shape,
                const std::vector<NumericType> & ext_data);

 virtual ~FunctorInitDat() = default;

 virtual const std::string name() const override
 {
  return "TensorFunctorInitDat";
 }

 virtual const std::string description() const override
 {
  return "Initializes a tensor with a given external data";
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

 TensorShape shape_;                      //shape of the full tensor
 std::vector<std::complex<double>> data_; //externally provided tensor body data (full tensor)
};


//DEFINITIONS:
template <typename NumericType>
FunctorInitDat::FunctorInitDat(const TensorShape & full_shape,
                               const std::vector<NumericType> & ext_data):
 shape_(full_shape), data_(ext_data.size())
{
 static_assert(std::is_same<NumericType,float>::value ||
               std::is_same<NumericType,double>::value ||
               std::is_same<NumericType,std::complex<float>>::value ||
               std::is_same<NumericType,std::complex<double>>::value,
               "#ERROR(exatn::numerics::FunctorInitDat): Invalid numeric data type!");
 assert(full_shape.getVolume() == ext_data.size());
 for(std::size_t i = 0; i < ext_data.size(); ++i) data_[i] = std::complex<double>(ext_data[i]);
}

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_FUNCTOR_INIT_DAT_HPP_
