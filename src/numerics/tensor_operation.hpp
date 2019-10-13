/** ExaTN::Numerics: Tensor operation
REVISION: 2019/10/13

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) A tensor operation is a formal numerical operation on one or more tensors.
**/

#ifndef EXATN_NUMERICS_TENSOR_OPERATION_HPP_
#define EXATN_NUMERICS_TENSOR_OPERATION_HPP_

#include "tensor_basic.hpp"
#include "tensor.hpp"

#include <memory>
#include <string>
#include <vector>
#include <complex>

#include <iostream>
#include <fstream>

namespace exatn{

namespace runtime{
 // Tensor operation execution handle:
 using TensorOpExecHandle = std::size_t;
 // Forward for TensorNodeExecutor:
 class TensorNodeExecutor;
}

namespace numerics{

class TensorOperation{ //abstract
public:

 /** Constructs a yet undefined tensor operation with
     the specified number of tensor/scalar arguments. **/
 TensorOperation(TensorOpCode opcode,       //tensor operation code
                 unsigned int num_operands, //required number of tensor operands
                 unsigned int num_scalars); //required number of scalar operands

 TensorOperation(const TensorOperation &) = default;
 TensorOperation & operator=(const TensorOperation &) = default;
 TensorOperation(TensorOperation &&) noexcept = default;
 TensorOperation & operator=(TensorOperation &&) noexcept = default;
 virtual ~TensorOperation() = default;

 /** Returns TRUE iff the tensor operation is fully set. **/
 virtual bool isSet() const = 0;

 /** Accepts tensor node executor (visitor pattern) which will actually
     execute the tensor operation in an asynchronous fashion, requiring
     subsequent synchronization via exec_handle. Returns an integer
     error code (0:Success). **/
 virtual int accept(runtime::TensorNodeExecutor & node_executor,
                    runtime::TensorOpExecHandle * exec_handle) = 0;

 /** Prints. **/
 virtual void printIt() const;
 virtual void printItFile(std::ofstream & output_file) const;

 /** Returns the tensor operation code (opcode). **/
 TensorOpCode getOpcode() const;

 /** Returns the number of tensor operands required for the tensor operation. **/
 unsigned int getNumOperands() const;

 /** Returns the number of tensor operands set. **/
 unsigned int getNumOperandsSet() const;

 /** Returns a unique integer tensor operand identifier. **/
 TensorHashType getTensorOperandHash(unsigned int op_num) const;

 /** Returns a co-owned pointer to a specific tensor operand, or nullptr if not yet set. **/
 std::shared_ptr<Tensor> getTensorOperand(unsigned int op_num,
                                          bool * conjugated = nullptr) const;

 /** Sets the next tensor operand. **/
 void setTensorOperand(std::shared_ptr<Tensor> tensor,
                       bool conjugated = false);

 /** Returns the number of scalar arguments required for the tensor operation. **/
 unsigned int getNumScalars() const;

 /** Returns the number of scalar arguments set explicitly. **/
 unsigned int getNumScalarsSet() const;

 /** Returns a specific scalar argument. **/
 std::complex<double> getScalar(unsigned int scalar_num) const;

 /** Sets a specific scalar argument. **/
 void setScalar(unsigned int scalar_num,
                const std::complex<double> scalar);

 /** Returns the symbolic tensor operation specification (index pattern). **/
 const std::string & getIndexPattern() const;

 /** Sets the symbolic tensor operation specification (index pattern).
     The tensor operation must have all its tensor/scalar operands set at this point.**/
 void setIndexPattern(const std::string & pattern);

 /** Sets the unique integer identifier of the tensor operation. **/
 void setId(std::size_t id);

 /** Returns the unique integer identifier of the tensor operation. **/
 std::size_t getId() const;

protected:

 std::string pattern_; //symbolic index pattern
 std::vector<std::pair<std::shared_ptr<Tensor>,bool>> operands_; //tensor operands (non-owning pointers)
 std::vector<std::complex<double>> scalars_; //additional scalars (prefactors)
 unsigned int num_operands_; //number of required tensor operands
 unsigned int num_scalars_; //number of required scalar arguments
 TensorOpCode opcode_; //tensor operation code
 std::size_t id_; //tensor operation id (unique integer identifier)
};

using createTensorOpFn = std::unique_ptr<TensorOperation> (*)(void);

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_OPERATION_HPP_
