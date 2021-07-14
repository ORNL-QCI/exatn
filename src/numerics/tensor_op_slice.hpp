/** ExaTN::Numerics: Tensor operation: Extracts a slice from a tensor
REVISION: 2021/07/13

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Extracts a slice from a tensor inside the processing backend:
     Operand 0 (slice) <= Operand 1
**/

#ifndef EXATN_NUMERICS_TENSOR_OP_SLICE_HPP_
#define EXATN_NUMERICS_TENSOR_OP_SLICE_HPP_

#include "tensor_basic.hpp"
#include "tensor_operation.hpp"

namespace exatn{

namespace numerics{

class TensorOpSlice: public TensorOperation{
public:

 TensorOpSlice();

 TensorOpSlice(const TensorOpSlice &) = default;
 TensorOpSlice & operator=(const TensorOpSlice &) = default;
 TensorOpSlice(TensorOpSlice &&) noexcept = default;
 TensorOpSlice & operator=(TensorOpSlice &&) noexcept = default;
 virtual ~TensorOpSlice() = default;

 virtual std::unique_ptr<TensorOperation> clone() const override{
  return std::unique_ptr<TensorOperation>(new TensorOpSlice(*this));
 }

 /** Returns TRUE iff the tensor operation is fully set. **/
 virtual bool isSet() const override;

 /** Accepts tensor node executor which will execute this tensor operation. **/
 virtual int accept(runtime::TensorNodeExecutor & node_executor,
                    runtime::TensorOpExecHandle * exec_handle) override;

 /** Decomposes a composite tensor operation into simple ones.
     Returns the total number of generated simple operations. **/
 virtual std::size_t decompose(std::function<bool (const Tensor &)> tensor_exists_locally) override;

 /** Create a new polymorphic instance of this subclass. **/
 static std::unique_ptr<TensorOperation> createNew();

private:

};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_OP_SLICE_HPP_
