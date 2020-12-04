/** ExaTN::Numerics: Tensor Functor: Prints a tensor to a file or standard output
REVISION: 2020/12/04

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (A) This tensor functor (method) is used to print a Tensor
     to a file or standard output with format:
      Storage format (string: {dense|list})
      Tensor name
      Tensor shape (space-separated dimension extents)
      Tensor signature (space-separated dimension base offsets)
      Tensor elements:
       Dense format: Numeric values (column-wise order), any number of values per line
       List format: Numeric value and Multi-index in each line
**/

#ifndef EXATN_NUMERICS_FUNCTOR_PRINT_HPP_
#define EXATN_NUMERICS_FUNCTOR_PRINT_HPP_

#include "Identifiable.hpp"

#include "tensor_basic.hpp"
#include "tensor_shape.hpp"

#include "tensor_method.hpp" //from TAL-SH

#include <string>
#include <complex>

#include "errors.hpp"

namespace exatn{

namespace numerics{

class FunctorPrint: public talsh::TensorFunctor<Identifiable>{
public:

 FunctorPrint() = default;
 FunctorPrint(const std::string & filename); //in: output file name

 virtual ~FunctorPrint() = default;

 virtual const std::string name() const override
 {
  return "TensorFunctorPrint";
 }

 virtual const std::string description() const override
 {
  return "Prints a tensor";
 }

 /** Packs data members into a byte packet. **/
 virtual void pack(BytePacket & packet) override;

 /** Unpacks data members from a byte packet. **/
 virtual void unpack(BytePacket & packet) override;

 /** Prints the local tensor slice to a file or standard output.
     Returns zero on success, or an error code otherwise.
     The talsh::Tensor slice is identified by its signature
     and shape that both can be accessed by talsh::Tensor methods. **/
 virtual int apply(talsh::Tensor & local_tensor) override;

private:

 std::string filename_; //file name (empty means standard output
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_FUNCTOR_PRINT_HPP_
