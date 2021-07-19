/** ExaTN::Numerics: Tensor operation: Destroys a tensor
REVISION: 2021/07/15

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Destroys a tensor inside the processing backend.
**/

#ifndef EXATN_NUMERICS_TENSOR_OP_DESTROY_HPP_
#define EXATN_NUMERICS_TENSOR_OP_DESTROY_HPP_

#include "tensor_basic.hpp"
#include "tensor_operation.hpp"

namespace exatn{

namespace numerics{

class TensorOpDestroy: public TensorOperation{
public:

 TensorOpDestroy();

 TensorOpDestroy(const TensorOpDestroy &) = default;
 TensorOpDestroy & operator=(const TensorOpDestroy &) = default;
 TensorOpDestroy(TensorOpDestroy &&) noexcept = default;
 TensorOpDestroy & operator=(TensorOpDestroy &&) noexcept = default;
 virtual ~TensorOpDestroy() = default;

 virtual std::unique_ptr<TensorOperation> clone() const override{
  return std::unique_ptr<TensorOperation>(new TensorOpDestroy(*this));
 }

 /** Returns TRUE iff the tensor operation is fully set. **/
 virtual bool isSet() const override;

 /** Accepts tensor node executor which will execute this tensor operation. **/
 virtual int accept(runtime::TensorNodeExecutor & node_executor,
                    runtime::TensorOpExecHandle * exec_handle) override;

 /** Decomposes a composite tensor operation into simple ones.
     Returns the total number of generated simple operations. **/
 virtual std::size_t decompose(const TensorMapper & tensor_mapper) override;

 /** Create a new polymorphic instance of this subclass. **/
 static std::unique_ptr<TensorOperation> createNew();

private:

};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_OP_DESTROY_HPP_
