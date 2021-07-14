/** ExaTN::Numerics: Tensor operation: All-reduces a tensor
REVISION: 2021/07/13

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) All-reduces a tensor inside the execution backend.
**/

#ifndef EXATN_NUMERICS_TENSOR_OP_ALLREDUCE_HPP_
#define EXATN_NUMERICS_TENSOR_OP_ALLREDUCE_HPP_

#include "tensor_basic.hpp"
#include "tensor_operation.hpp"

#include "mpi_proxy.hpp"

namespace exatn{

namespace numerics{

class TensorOpAllreduce: public TensorOperation{
public:

 TensorOpAllreduce();

 TensorOpAllreduce(const TensorOpAllreduce &) = default;
 TensorOpAllreduce & operator=(const TensorOpAllreduce &) = default;
 TensorOpAllreduce(TensorOpAllreduce &&) noexcept = default;
 TensorOpAllreduce & operator=(TensorOpAllreduce &&) noexcept = default;
 virtual ~TensorOpAllreduce() = default;

 virtual std::unique_ptr<TensorOperation> clone() const override{
  return std::unique_ptr<TensorOperation>(new TensorOpAllreduce(*this));
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

 /** Resets the MPI communicator. **/
 bool resetMPICommunicator(const MPICommProxy & intra_comm);

 /** Returns the MPI communicator. **/
 const MPICommProxy & getMPICommunicator() const;

private:

 MPICommProxy intra_comm_; //MPI intra-communicator
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_OP_ALLREDUCE_HPP_
