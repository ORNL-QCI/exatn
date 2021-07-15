/** ExaTN::Numerics: Tensor operation: Decomposes a tensor into three tensor factors via SVD
REVISION: 2021/07/15

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Decomposes a tensor into three tensor factors via SVD, for example:
     D(a,b,c,d,e) = L(c,i,e,j) * S(i,j)    * R(d,j,a,b,i)
     Operand 3    = Operand 0  * Operand 2 * Operand 1
     Note that the ordering of the contracted indices is not guaranteed.
**/

#ifndef EXATN_NUMERICS_TENSOR_OP_DECOMPOSE_SVD3_HPP_
#define EXATN_NUMERICS_TENSOR_OP_DECOMPOSE_SVD3_HPP_

#include "tensor_basic.hpp"
#include "tensor_operation.hpp"

namespace exatn{

namespace numerics{

class TensorOpDecomposeSVD3: public TensorOperation{
public:

 TensorOpDecomposeSVD3();

 TensorOpDecomposeSVD3(const TensorOpDecomposeSVD3 &) = default;
 TensorOpDecomposeSVD3 & operator=(const TensorOpDecomposeSVD3 &) = default;
 TensorOpDecomposeSVD3(TensorOpDecomposeSVD3 &&) noexcept = default;
 TensorOpDecomposeSVD3 & operator=(TensorOpDecomposeSVD3 &&) noexcept = default;
 virtual ~TensorOpDecomposeSVD3() = default;

 virtual std::unique_ptr<TensorOperation> clone() const override{
  return std::unique_ptr<TensorOperation>(new TensorOpDecomposeSVD3(*this));
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

 /** Resets the absorption mode for the singular values factor: {'N','L','R','S'} **/
 bool resetAbsorptionMode(const char absorb_mode = 'N');

private:

 char absorb_singular_values_; //regulates the absorption of the singular values factor
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_OP_DECOMPOSE_SVD3_HPP_
