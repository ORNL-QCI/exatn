/** ExaTN::Numerics: Tensor Functor: Initialization from a file
REVISION: 2020/12/03

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (A) This tensor functor (method) is used to initialize a Tensor
     with some externally provided data read from a file with format:
      Storage format (string: {dense|list})
      Tensor name
      Tensor shape (space-separated dimension extents)
      Tensor signature (space-separated dimension base offsets)
      Tensor elements:
       Dense format: Numeric values (column-wise order), any number of values per line
       List format: Numeric value and Multi-index in each line
**/

#ifndef EXATN_NUMERICS_FUNCTOR_INIT_FILE_HPP_
#define EXATN_NUMERICS_FUNCTOR_INIT_FILE_HPP_

#include "Identifiable.hpp"

#include "tensor_basic.hpp"
#include "tensor_shape.hpp"

#include "tensor_method.hpp" //from TAL-SH

#include <string>
#include <complex>

#include "errors.hpp"

namespace exatn{

namespace numerics{

class FunctorInitFile: public talsh::TensorFunctor<Identifiable>{
public:

 FunctorInitFile(const std::string & filename); //in: file name which contains tensor data

 virtual ~FunctorInitFile() = default;

 virtual const std::string name() const override
 {
  return "TensorFunctorInitFile";
 }

 virtual const std::string description() const override
 {
  return "Initializes a tensor with a given external data read from a file";
 }

 /** Packs data members into a byte packet. **/
 virtual void pack(BytePacket & packet) override;

 /** Unpacks data members from a byte packet. **/
 virtual void unpack(BytePacket & packet) override;

 /** Initializes the local tensor slice with external data
     read from a file. Returns zero on success, or an error code
     otherwise. The talsh::Tensor slice is identified by its signature
     and shape that both can be accessed by talsh::Tensor methods. **/
 virtual int apply(talsh::Tensor & local_tensor) override;

private:

 std::string filename_; //file name which contains tensor data
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_FUNCTOR_INIT_FILE_HPP_
