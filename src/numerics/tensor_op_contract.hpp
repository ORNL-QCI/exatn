/** ExaTN::Numerics: Tensor operation: Contracts two tensors and accumulates the result into another tensor
REVISION: 2020/06/06

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Contracts two tensors and accumulates the result into another tensor
     inside the processing backend:
     Operand 0 += Operand 1 * Operand 2 * prefactor
**/

#ifndef EXATN_NUMERICS_TENSOR_OP_CONTRACT_HPP_
#define EXATN_NUMERICS_TENSOR_OP_CONTRACT_HPP_

#include "tensor_basic.hpp"
#include "tensor_operation.hpp"

namespace exatn{

namespace numerics{

class TensorOpContract: public TensorOperation{
public:

 TensorOpContract();

 TensorOpContract(const TensorOpContract &) = default;
 TensorOpContract & operator=(const TensorOpContract &) = default;
 TensorOpContract(TensorOpContract &&) noexcept = default;
 TensorOpContract & operator=(TensorOpContract &&) noexcept = default;
 virtual ~TensorOpContract() = default;

 virtual std::unique_ptr<TensorOperation> clone() const override{
  return std::unique_ptr<TensorOperation>(new TensorOpContract(*this));
 }

 /** Returns TRUE iff the tensor operation is fully set. **/
 virtual bool isSet() const override;

 /** Accepts tensor node executor which will execute this tensor operation. **/
 virtual int accept(runtime::TensorNodeExecutor & node_executor,
                    runtime::TensorOpExecHandle * exec_handle) override;

 /** Returns the flop estimate for the tensor operation. **/
 virtual double getFlopEstimate() const override;

 /** Create a new polymorphic instance of this subclass. **/
 static std::unique_ptr<TensorOperation> createNew();

private:

};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_OP_CONTRACT_HPP_
