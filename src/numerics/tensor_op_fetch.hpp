/** ExaTN::Numerics: Tensor operation: Fetches remote tensor data
REVISION: 2021/07/13

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Fetches tensor data from a remote MPI process.
**/

#ifndef EXATN_NUMERICS_TENSOR_OP_FETCH_HPP_
#define EXATN_NUMERICS_TENSOR_OP_FETCH_HPP_

#include "tensor_basic.hpp"
#include "tensor_operation.hpp"

#include "mpi_proxy.hpp"

namespace exatn{

namespace numerics{

class TensorOpFetch: public TensorOperation{
public:

 TensorOpFetch();

 TensorOpFetch(const TensorOpFetch &) = default;
 TensorOpFetch & operator=(const TensorOpFetch &) = default;
 TensorOpFetch(TensorOpFetch &&) noexcept = default;
 TensorOpFetch & operator=(TensorOpFetch &&) noexcept = default;
 virtual ~TensorOpFetch() = default;

 virtual std::unique_ptr<TensorOperation> clone() const override{
  return std::unique_ptr<TensorOperation>(new TensorOpFetch(*this));
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

 /** Resets the remote process rank. **/
 bool resetRemoteProcessRank(unsigned int rank);

 /** Returns the remote process rank. **/
 int getRemoteProcessRank() const;

 /** Resets the MPI message tag. **/
 bool resetMessageTag(int tag);

 /** Returns the MPI message tag. **/
 int getMessageTag() const;

private:

 MPICommProxy intra_comm_; //MPI intra-communicator
 int remote_rank_; //MPI process rank
 int message_tag_; //MPI message tag
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_OP_FETCH_HPP_
