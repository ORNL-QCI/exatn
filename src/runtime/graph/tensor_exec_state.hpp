/** ExaTN:: Tensor Runtime: Tensor graph execution state
REVISION: 2019/07/29

Copyright (C) 2018-2019 Dmitry Lyakh, Tiffany Mintz, Alex McCaskey
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle)

Rationale:
 (a) Tensor graph is a directed acyclic graph in which vertices
     represent tensor operations and directed edges represent
     dependencies between them: A directed edge from node1 to
     node2 indicates that node1 depends on node2. Each DAG node
     has its unique integer vertex id (VertexIdType) returned
     when the node is added to the DAG.
 (b) The tensor graph contains:
     1. The DAG implementation (in the DirectedBoostGraph subclass);
     2. The DAG execution state (TensorExecState data member).
 (c) The execution state of each Tensor is either of the following:
     1. None (no outstanding reads or writes on the Tensor);
     2. Read (one or more most recently submitted tensor operations
        involving the Tensor perform a read on it). This is the READ
        epoch characterized by a positive integer equal to the number
        of outstanding reads on the Tensor in the current (read) epoch.
     3. Write (most recent tensor operation on the Tensor is a write).
        This is the WRITE epoch characterized a negative integer -1
        denoting a single outstanding write on the Tensor in the
        current (write) epoch.
**/

#ifndef EXATN_RUNTIME_TENSOR_EXEC_STATE_HPP_
#define EXATN_RUNTIME_TENSOR_EXEC_STATE_HPP_

#include "tensor_operation.hpp"
#include "tensor.hpp"

#include <unordered_map>
#include <list>
#include <memory>

namespace exatn {
namespace runtime {

// Tensor Graph node id (DirectedBoostGraph vertex descriptor):
using VertexIdType = std::size_t; //must match with boost::graph vertex descriptor type

// Tensor implementation:
using numerics::TensorHashType; //each numerics::Tensor has its unique integer hash
using numerics::Tensor;
using numerics::TensorOperation;


class TensorExecState {

protected:

  struct TensorExecInfo {
    std::size_t update_count; //total number of outstanding updates on a given Tensor in the current DAG
    int rw_epoch; //>0: number of current epoch reads; -1: current epoch write (single)
    std::vector<VertexIdType> rw_epoch_nodes; //nodes participating in the current R/W epoch

    TensorExecInfo(): update_count(0), rw_epoch(0) {}
    TensorExecInfo(const TensorExecInfo &) = default;
    TensorExecInfo & operator=(const TensorExecInfo &) = default;
    TensorExecInfo(TensorExecInfo &&) noexcept = default;
    TensorExecInfo & operator=(TensorExecInfo &&) noexcept = default;
    ~TensorExecInfo() = default;
  };

public:
  TensorExecState() = default;
  TensorExecState(const TensorExecState &) = delete;
  TensorExecState & operator=(const TensorExecState &) = delete;
  TensorExecState(TensorExecState &&) noexcept = default;
  TensorExecState & operator=(TensorExecState &&) noexcept = default;
  ~TensorExecState() = default;

  /** Returns the list of nodes participating in the current R/W epoch:
      epoch > 0: This is the number of reads in the current Read epoch;
      epoch = -1: This is a single write in the current Write epoch. **/
  const std::vector<VertexIdType> * getTensorEpochNodes(const Tensor & tensor,
                                                        int * epoch);
  /** Registers a new read on a Tensor. Returns the current epoch. **/
  int registerTensorRead(const Tensor & tensor,
                         VertexIdType node_id);
  /** Registers a new write on a Tensor. Returns the current epoch. **/
  int registerTensorWrite(const Tensor & tensor,
                          VertexIdType node_id);
  /** Registers a completion of an outstanding write on a Tensor.
      Returns the updated outstanding update count on the Tensor. **/
  std::size_t registerWriteCompletion(const Tensor & tensor);
  /** Returns the current outstanding update count on the tensor in the DAG. **/
  std::size_t getTensorUpdateCount(const Tensor & tensor);

  /** Registers a DAG node without dependencies. **/
  void registerDependencyFreeNode(VertexIdType node_id);
  /** Extracts a dependency-free node from the list.
      Returns FALSE if no such node exists. **/
  bool extractDependencyFreeNode(VertexIdType * node_id);

  /** Registers a DAG node as being executed. **/
  void registerExecutingNode(VertexIdType node_id);
  /** Extracts an executed DAG node from the list. **/
  bool extractExecutingNode(VertexIdType * node_id);

private:
  /** Table for tracking the execution status of a given tensor:
      Tensor Hash --> TensorExecInfo **/
  std::unordered_map<TensorHashType,TensorExecInfo> tensor_info_;
  /** List of dependency-free unexecuted DAG nodes **/
  std::list<VertexIdType> nodes_ready_;
  /** List of the currently executed DAG nodes **/
  std::list<VertexIdType> nodes_executing_;
};

} // namespace runtime
} // namespace exatn

#endif //EXATN_RUNTIME_TENSOR_EXEC_STATE_HPP_
