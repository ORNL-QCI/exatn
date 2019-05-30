/** ExaTN::Numerics: Tensor operation: Adds a tensor to another tensor
REVISION: 2019/05/30

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Adds a tensor to another tensor inside the processing backend.
**/

#ifndef EXATN_NUMERICS_TENSOR_OP_ADD_HPP_
#define EXATN_NUMERICS_TENSOR_OP_ADD_HPP_

#include "tensor_basic.hpp"
#include "tensor_operation.hpp"

namespace exatn{

namespace numerics{

class TensorOpAdd: public TensorOperation{
public:

 TensorOpAdd();

 TensorOpAdd(const TensorOpAdd &) = default;
 TensorOpAdd & operator=(const TensorOpAdd &) = default;
 TensorOpAdd(TensorOpAdd &&) noexcept = default;
 TensorOpAdd & operator=(TensorOpAdd &&) noexcept = default;
 virtual ~TensorOpAdd() = default;

 /** Returns TRUE iff the tensor operation is fully set. **/
 virtual bool isSet() const override;

private:

};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_OP_ADD_HPP_
