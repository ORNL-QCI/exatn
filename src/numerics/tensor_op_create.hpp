/** ExaTN::Numerics: Tensor operation: Creates a tensor
REVISION: 2019/06/03

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Creates a tensor inside the processing backend.
**/

#ifndef EXATN_NUMERICS_TENSOR_OP_CREATE_HPP_
#define EXATN_NUMERICS_TENSOR_OP_CREATE_HPP_

#include "tensor_basic.hpp"
#include "tensor_operation.hpp"

namespace exatn{

namespace numerics{

class TensorOpCreate: public TensorOperation{
public:

 TensorOpCreate();

 TensorOpCreate(const TensorOpCreate &) = default;
 TensorOpCreate & operator=(const TensorOpCreate &) = default;
 TensorOpCreate(TensorOpCreate &&) noexcept = default;
 TensorOpCreate & operator=(TensorOpCreate &&) noexcept = default;
 virtual ~TensorOpCreate() = default;

 /** Returns TRUE iff the tensor operation is fully set. **/
 virtual bool isSet() const override;

 /** Create a new polymorphic instance of this subclass. **/
 static std::unique_ptr<TensorOperation> createNew();

private:

};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_OP_CREATE_HPP_
