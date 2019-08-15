/** ExaTN::Numerics: Tensor operation: Transforms/initializes a tensor
REVISION: 2019/08/15

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Transforms/initializes a tensor inside the processing backend.
**/

#ifndef EXATN_NUMERICS_TENSOR_OP_TRANSFORM_HPP_
#define EXATN_NUMERICS_TENSOR_OP_TRANSFORM_HPP_

#include "tensor_basic.hpp"
#include "tensor_operation.hpp"

namespace exatn{

namespace numerics{

class TensorOpTransform: public TensorOperation{
public:

 TensorOpTransform();

 TensorOpTransform(const TensorOpTransform &) = default;
 TensorOpTransform & operator=(const TensorOpTransform &) = default;
 TensorOpTransform(TensorOpTransform &&) noexcept = default;
 TensorOpTransform & operator=(TensorOpTransform &&) noexcept = default;
 virtual ~TensorOpTransform() = default;

 /** Returns TRUE iff the tensor operation is fully set. **/
 virtual bool isSet() const override;

 /** Accepts tensor node executor which will execute this tensor operation. **/
 virtual void accept(runtime::TensorNodeExecutor & node_executor) override;

 /** Create a new polymorphic instance of this subclass. **/
 static std::unique_ptr<TensorOperation> createNew();

private:

};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_OP_TRANSFORM_HPP_
