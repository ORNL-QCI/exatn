/** ExaTN::Numerics: Tensor operation: Orthogonalizes a tensor via MGS
REVISION: 2021/07/13

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Orthogonalizes a tensor via MGS.
**/

#ifndef EXATN_NUMERICS_TENSOR_OP_ORTHOGONALIZE_MGS_HPP_
#define EXATN_NUMERICS_TENSOR_OP_ORTHOGONALIZE_MGS_HPP_

#include "tensor_basic.hpp"
#include "tensor_operation.hpp"

namespace exatn{

namespace numerics{

class TensorOpOrthogonalizeMGS: public TensorOperation{
public:

 TensorOpOrthogonalizeMGS();

 TensorOpOrthogonalizeMGS(const TensorOpOrthogonalizeMGS &) = default;
 TensorOpOrthogonalizeMGS & operator=(const TensorOpOrthogonalizeMGS &) = default;
 TensorOpOrthogonalizeMGS(TensorOpOrthogonalizeMGS &&) noexcept = default;
 TensorOpOrthogonalizeMGS & operator=(TensorOpOrthogonalizeMGS &&) noexcept = default;
 virtual ~TensorOpOrthogonalizeMGS() = default;

 virtual std::unique_ptr<TensorOperation> clone() const override{
  return std::unique_ptr<TensorOperation>(new TensorOpOrthogonalizeMGS(*this));
 }

 /** Returns TRUE iff the tensor operation is fully set. **/
 virtual bool isSet() const override;

 /** Accepts tensor node executor which will execute this tensor operation. **/
 virtual int accept(runtime::TensorNodeExecutor & node_executor,
                    runtime::TensorOpExecHandle * exec_handle) override;

 /** Decomposes a composite tensor operation into simple ones.
     Returns the total number of generated simple operations. **/
 virtual std::size_t decompose(std::function<bool (const Tensor &)> tensor_exists_locally) override
 {
  return 0;
 }

 /** Create a new polymorphic instance of this subclass. **/
 static std::unique_ptr<TensorOperation> createNew();

private:

};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_OP_ORTHOGONALIZE_MGS_HPP_
