/** ExaTN::Numerics: Tensor operation: Inserts a slice into a tensor
REVISION: 2020/04/20

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Inserts a slice into a tensor inside the processing backend:
     Operand 0 <= Operand 1 (slice)
**/

#ifndef EXATN_NUMERICS_TENSOR_OP_INSERT_HPP_
#define EXATN_NUMERICS_TENSOR_OP_INSERT_HPP_

#include "tensor_basic.hpp"
#include "tensor_operation.hpp"

namespace exatn{

namespace numerics{

class TensorOpInsert: public TensorOperation{
public:

 TensorOpInsert();

 TensorOpInsert(const TensorOpInsert &) = default;
 TensorOpInsert & operator=(const TensorOpInsert &) = default;
 TensorOpInsert(TensorOpInsert &&) noexcept = default;
 TensorOpInsert & operator=(TensorOpInsert &&) noexcept = default;
 virtual ~TensorOpInsert() = default;

 /** Returns TRUE iff the tensor operation is fully set. **/
 virtual bool isSet() const override;

 /** Accepts tensor node executor which will execute this tensor operation. **/
 virtual int accept(runtime::TensorNodeExecutor & node_executor,
                    runtime::TensorOpExecHandle * exec_handle) override;

 /** Create a new polymorphic instance of this subclass. **/
 static std::unique_ptr<TensorOperation> createNew();

private:

};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_OP_INSERT_HPP_
