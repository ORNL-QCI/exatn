/** ExaTN::Numerics: Tensor operation
REVISION: 2019/05/30

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) A tensor operation is a formal numerical operation on one or more tensors.
**/

#ifndef EXATN_NUMERICS_TENSOR_OPERATION_HPP_
#define EXATN_NUMERICS_TENSOR_OPERATION_HPP_

#include "tensor_basic.hpp"
#include "tensor.hpp"

#include <string>
#include <vector>
#include <complex>

namespace exatn{

namespace numerics{

class TensorOperation{
public:

 /** Constructs a yet undefined tensor operation with
     the specified number of tensor/scalar arguments. **/
 TensorOperation(unsigned int num_operands, //required number of tensor operands
                 unsigned int num_scalars); //required number of scalar operands

 TensorOperation(const TensorOperation &) = default;
 TensorOperation & operator=(const TensorOperation &) = default;
 TensorOperation(TensorOperation &&) noexcept = default;
 TensorOperation & operator=(TensorOperation &&) noexcept = default;
 virtual ~TensorOperation() = default;

 /** Returns TRUE iff the tensor operation is fully set. **/
 virtual bool isSet() const = 0;

 /** Returns the number of tensor operands required for the tensor operation. **/
 unsigned int getNumOperands() const;

 /** Returns the number of tensor operands set. **/
 unsigned int getNumOperandsSet() const;

 /** Returns a non-owning pointer to a specific tensor operand, or nullptr if not set. **/
 const Tensor * getTensorOperand(unsigned int op_num) const;

 /** Sets the next tensor operand. **/
 void setTensorOperand(const Tensor * tensor);

 /** Returns the number of scalar arguments required for the tensor operation. **/
 unsigned int getNumScalars() const;

 /** Returns the number of scalar arguments set explicitly. **/
 unsigned int getNumScalarsSet() const;

 /** Returns a specific scalar argument. **/
 std::complex<double> getScalar(unsigned int scalar_num) const;

 /** Sets the next scalar argument. **/
 void setScalar(unsigned int scalar_num,
                const std::complex<double> scalar);

 /** Returns the symbolic tensor operation specification (index pattern). **/
 const std::string & getIndexPattern() const;

 /** Sets the symbolic tensor operation specification (index pattern).
     The tensor operation must have all its tensor/scalar operands set at this point.**/
 void setIndexPattern(const std::string & pattern);

protected:

 std::string pattern_; //symbolic index pattern
 std::vector<const Tensor *> operands_; //tensor operands (non-owning pointers)
 std::vector<std::complex<double>> scalars_; //additional scalars (prefactors)
 unsigned int num_operands_; //number of required tensor operands
 unsigned int num_scalars_; //number of required scalar arguments

};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_OPERATION_HPP_
