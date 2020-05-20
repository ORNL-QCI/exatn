/** ExaTN::Numerics: Tensor operation
REVISION: 2020/05/20

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) A tensor operation is a formal numerical operation on one or more tensors.
 (b) A tensor operation may have mutable (output) and immutable (input) tensor operands.
     The mutable tensor operands must always precede immutable tensor operands!
**/

#ifndef EXATN_NUMERICS_TENSOR_OPERATION_HPP_
#define EXATN_NUMERICS_TENSOR_OPERATION_HPP_

#include "tensor_basic.hpp"
#include "tensor.hpp"
#include "timers.hpp"

#include <tuple>
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
 TensorOperation(TensorOpCode opcode,       //in: tensor operation code
                 unsigned int num_operands, //in: required number of tensor operands
                 unsigned int num_scalars,  //in: required number of scalar operands
                 std::size_t mutability);   //in: bit-mask for operand mutability

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

 /** Returns the number of output tensor operands required for the tensor operation. **/
 unsigned int getNumOperandsOut() const;

 /** Returns the number of tensor operands set. **/
 unsigned int getNumOperandsSet() const;

 /** Returns a unique integer tensor operand identifier. **/
 TensorHashType getTensorOperandHash(unsigned int op_num) const;

 /** Returns the complex conjugation status of a tensor operand
     (whether or not the operand enters the operation as complex conjugated). **/
 bool operandIsConjugated(unsigned int op_num) const;

 /** Returns the mutation status of a tensor operand
     (whether or not the operand is mutated during the tensor operation). **/
 bool operandIsMutable(unsigned int op_num) const;

 /** Returns a co-owned pointer to a specific tensor operand, or nullptr if not yet set. **/
 std::shared_ptr<Tensor> getTensorOperand(unsigned int op_num,             //in: operand position
                                          bool * conjugated = nullptr,     //out: complex conjugation status
                                          bool * mutated = nullptr) const; //out: mutability status

 /** Sets the next tensor operand. **/
 void setTensorOperand(std::shared_ptr<Tensor> tensor, //in: tensor
                       bool conjugated = false);       //in: complex conjugation status

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
     The tensor operation must have all its tensor/scalar operands set at this point.
     It is allowed to reset an already set index pattern via this function. **/
 void setIndexPattern(const std::string & pattern);

 /** Sets the unique integer identifier of the tensor operation. **/
 void setId(std::size_t id);

 /** Returns the unique integer identifier of the tensor operation. **/
 std::size_t getId() const;

 /** Returns a unique integerhash for the tensor operation. **/
 TensorHashType getTensorOpHash() const;

 /** Records the start time stamp for tensor operation execution. **/
 inline bool recordStartTime(){
  return timer_.start();
 }

 /** Records the finish time stamp for tensor operation execution. **/
 inline bool recordFinishTime(){
  return timer_.stop();
 }

 /** Retrieves the tensor operation duration between the finish and start time stamps. **/
 inline double getTimeDuration() const{
  return timer_.getDuration();
 }

 /** Returns the start time of the tensor operation execution. **/
 inline double getStartTime() const{
  return timer_.getStartTime();
 }

 /** Returns the finish time of the tensor operation execution. **/
 inline double getFinishTime() const{
  return timer_.getFinishTime();
 }

private:

 /** Sets the next tensor operand with its mutability status. **/
 void setTensorOperand(std::shared_ptr<Tensor> tensor, //in: tensor
                       bool conjugated,                //in: complex conjugation status
                       bool mutated);                  //in: mutability status

protected:

 std::string pattern_; //symbolic index pattern
 std::vector<std::tuple<std::shared_ptr<Tensor>,bool,bool>> operands_; //tensor operands <operand,conjugation,mutation>
 std::vector<std::complex<double>> scalars_; //additional scalars (prefactors)
 unsigned int num_operands_; //number of required tensor operands
 unsigned int num_scalars_; //number of required scalar arguments
 std::size_t mutation_; //default operand mutability bits: Bit X --> Operand #X
 TensorOpCode opcode_; //tensor operation code
 std::size_t id_; //tensor operation id (unique integer identifier)
 Timer timer_; //internal timer
};

using createTensorOpFn = std::unique_ptr<TensorOperation> (*)(void);

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_OPERATION_HPP_
