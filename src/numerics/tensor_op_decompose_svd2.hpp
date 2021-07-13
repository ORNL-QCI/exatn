/** ExaTN::Numerics: Tensor operation: Decomposes a tensor into two tensor factors via SVD
REVISION: 2021/07/13

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Decomposes a tensor into two tensor factors via SVD, for example:
     D(a,b,c,d,e) = L(c,i,e,j) * R(d,j,a,b,i)
     Operand 2    = Operand 0  * Operand 1
**/

#ifndef EXATN_NUMERICS_TENSOR_OP_DECOMPOSE_SVD2_HPP_
#define EXATN_NUMERICS_TENSOR_OP_DECOMPOSE_SVD2_HPP_

#include "tensor_basic.hpp"
#include "tensor_operation.hpp"

namespace exatn{

namespace numerics{

class TensorOpDecomposeSVD2: public TensorOperation{
public:

 TensorOpDecomposeSVD2();

 TensorOpDecomposeSVD2(const TensorOpDecomposeSVD2 &) = default;
 TensorOpDecomposeSVD2 & operator=(const TensorOpDecomposeSVD2 &) = default;
 TensorOpDecomposeSVD2(TensorOpDecomposeSVD2 &&) noexcept = default;
 TensorOpDecomposeSVD2 & operator=(TensorOpDecomposeSVD2 &&) noexcept = default;
 virtual ~TensorOpDecomposeSVD2() = default;

 virtual std::unique_ptr<TensorOperation> clone() const override{
  return std::unique_ptr<TensorOperation>(new TensorOpDecomposeSVD2(*this));
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

#endif //EXATN_NUMERICS_TENSOR_OP_DECOMPOSE_SVD2_HPP_
