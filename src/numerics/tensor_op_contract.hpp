/** ExaTN::Numerics: Tensor operation: Contracts two tensors and accumulates the result into another tensor
REVISION: 2019/05/30

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Contracts two tensors and accumulates the result into another tensor
     inside the processing backend.
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

 /** Returns TRUE iff the tensor operation is fully set. **/
 virtual bool isSet() const override;

private:

};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_OP_CONTRACT_HPP_
