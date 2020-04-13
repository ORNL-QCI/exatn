/** ExaTN::Numerics: Tensor operation factory
REVISION: 2020/04/13

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Creates new tensor operations of desired kind.
**/

#ifndef EXATN_NUMERICS_TENSOR_OP_FACTORY_HPP_
#define EXATN_NUMERICS_TENSOR_OP_FACTORY_HPP_

#include "tensor_basic.hpp"
#include "tensor_operation.hpp"
#include "tensor_op_create.hpp"
#include "tensor_op_destroy.hpp"
#include "tensor_op_transform.hpp"
#include "tensor_op_slice.hpp"
#include "tensor_op_insert.hpp"
#include "tensor_op_add.hpp"
#include "tensor_op_contract.hpp"
#include "tensor_op_decompose_svd3.hpp"
#include "tensor_op_decompose_svd2.hpp"
#include "tensor_op_orthogonalize_svd.hpp"
#include "tensor_op_orthogonalize_mgs.hpp"
#include "tensor_op_broadcast.hpp"
#include "tensor_op_allreduce.hpp"

#include <memory>
#include <map>

namespace exatn{

namespace numerics{

class TensorOpFactory{
public:

 TensorOpFactory(const TensorOpFactory &) = delete;
 TensorOpFactory & operator=(const TensorOpFactory &) = delete;
 TensorOpFactory(TensorOpFactory &&) noexcept = default;
 TensorOpFactory & operator=(TensorOpFactory &&) noexcept = default;
 ~TensorOpFactory() = default;

 /** Registers a new tensor operation subtype to produce instances of. **/
 void registerTensorOp(TensorOpCode opcode, createTensorOpFn creator);

 /** Creates a new instance of a desired subtype. **/
 std::unique_ptr<TensorOperation> createTensorOp(TensorOpCode opcode);
 /** Creates a new instance of a desired subtype. **/
 std::shared_ptr<TensorOperation> createTensorOpShared(TensorOpCode opcode);

 /** Returns a pointer to the TensorOpFactory singleton. **/
 static TensorOpFactory * get();

private:

 TensorOpFactory(); //private ctor

 std::map<TensorOpCode,createTensorOpFn> factory_map_;

};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_OP_FACTORY_HPP_
