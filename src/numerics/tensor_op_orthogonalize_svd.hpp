/** ExaTN::Numerics: Tensor operation: Orthogonalizes a tensor via SVD
REVISION: 2020/05/22

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Orthogonalizes a tensor via SVD, for example:
     D(a,b,c,d,e) := L(c,i,e,j) * R(d,j,a,b,i)
     where tensors L(c,i,e,j) and R(d,j,a,b,i) come from the SVD decomposition:
     D(a,b,c,d,e) = L(c,i,e,j) * S(i,j) * R(d,j,a,b,i)
**/

#ifndef EXATN_NUMERICS_TENSOR_OP_ORTHOGONALIZE_SVD_HPP_
#define EXATN_NUMERICS_TENSOR_OP_ORTHOGONALIZE_SVD_HPP_

#include "tensor_basic.hpp"
#include "tensor_operation.hpp"

namespace exatn{

namespace numerics{

class TensorOpOrthogonalizeSVD: public TensorOperation{
public:

 TensorOpOrthogonalizeSVD();

 TensorOpOrthogonalizeSVD(const TensorOpOrthogonalizeSVD &) = default;
 TensorOpOrthogonalizeSVD & operator=(const TensorOpOrthogonalizeSVD &) = default;
 TensorOpOrthogonalizeSVD(TensorOpOrthogonalizeSVD &&) noexcept = default;
 TensorOpOrthogonalizeSVD & operator=(TensorOpOrthogonalizeSVD &&) noexcept = default;
 virtual ~TensorOpOrthogonalizeSVD() = default;

 virtual std::unique_ptr<TensorOperation> clone() const override{
  return std::unique_ptr<TensorOperation>(new TensorOpOrthogonalizeSVD(*this));
 }

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

#endif //EXATN_NUMERICS_TENSOR_OP_ORTHOGONALIZE_SVD_HPP_
