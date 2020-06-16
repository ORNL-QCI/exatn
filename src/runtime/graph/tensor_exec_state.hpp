/** ExaTN:: Tensor Runtime: Tensor graph execution state
REVISION: 2020/06/16

Copyright (C) 2018-2020 Dmitry Lyakh, Tiffany Mintz, Alex McCaskey
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle)

Rationale:
 (a) Tensor graph is a directed acyclic graph in which vertices
     represent tensor operations and directed edges represent
     dependencies between them: A directed edge from node1 to
     node2 indicates that node1 depends on node2. Each DAG node
     has its unique integer vertex id (VertexIdType) returned
     when the node is appended into the DAG.
 (b) The tensor graph contains:
     1. The DAG implementation (DirectedBoostGraph subclass);
     2. The DAG execution state (TensorExecState data member).
 (c) The execution state of each Tensor in the DAG is either of the following:
     1. None (no outstanding reads or writes on the Tensor);
     2. Read (one or more most recently submitted tensor operations
        involving the Tensor perform reads on it). This is the READ
        epoch characterized by a positive integer equal to the number
        of outstanding reads on the Tensor in the current (read) epoch.
     3. Write (most recent tensor operation on the Tensor is a write).
        This is the WRITE epoch characterized by a negative integer -1
        denoting the single outstanding write on the Tensor in the
        current (write) epoch.
     The execution state of a Tensor is progressing through alternating
     read and write epochs, introducing read-after-write, write-after-write,
     and write-after-read dependencies between tensor nodes with stored
     tensor operations operating on the same data (Tensor). Importantly,
     the execution state of a Tensor is defined with respect to the DAG
     builder, that is, every time a new tensor operation is added into
     the DAG the execution state of each participating tensor is inspected
     and possibly altered (switched to another epoch). Thus, the execution
     state of a tensor is only used for establishing data dependencies for
     newly added DAG nodes, it has nothing to do with actual DAG execution.
**/

#ifndef EXATN_RUNTIME_TENSOR_EXEC_STATE_HPP_
#define EXATN_RUNTIME_TENSOR_EXEC_STATE_HPP_

#include "tensor_operation.hpp"
#include "tensor.hpp"

#include <unordered_map>
#include <list>
#include <memory>
#include <atomic>

namespace exatn {
namespace runtime {

// Tensor Graph node id (DirectedBoostGraph vertex descriptor):
using VertexIdType = std::size_t; //must match with boost::graph vertex descriptor type

// Tensor implementation:
using numerics::TensorHashType; //each numerics::Tensor has its unique integer hash
using numerics::Tensor;
using numerics::TensorOperation;

using ExecutingNodesIterator = typename std::list<std::pair<VertexIdType,TensorOpExecHandle>>::const_iterator;


class TensorExecState {

protected:

  struct TensorExecInfo {
    std::atomic<std::size_t> update_count;    //total number of outstanding updates on a given Tensor in the current DAG
    std::atomic<int> rw_epoch;                //>0: number of current epoch reads; -1: current epoch write (single)
    std::vector<VertexIdType> rw_epoch_nodes; //nodes participating in the current R/W epoch (either read or write)

    TensorExecInfo(): update_count(0), rw_epoch(0) {}
    TensorExecInfo(const TensorExecInfo &) = delete;
    TensorExecInfo & operator=(const TensorExecInfo &) = delete;
    TensorExecInfo(TensorExecInfo &&) noexcept = delete;
    TensorExecInfo & operator=(TensorExecInfo &&) noexcept = delete;
    ~TensorExecInfo() = default;
  };

public:

  TensorExecState(): front_node_(0) {}

  TensorExecState(const TensorExecState &) = delete;
  TensorExecState & operator=(const TensorExecState &) = delete;
  TensorExecState(TensorExecState &&) noexcept = default;
  TensorExecState & operator=(TensorExecState &&) noexcept = default;
  ~TensorExecState() = default;

  /** Returns the list of nodes participating in the current R/W epoch:
      epoch > 0: This is the number of reads in the current Read epoch;
      epoch = -1: This is the single write in the current Write epoch. **/
  const std::vector<VertexIdType> * getTensorEpochNodes(const Tensor & tensor,
                                                        int * epoch);
  /** Registers a new read on a Tensor. Returns the current epoch R/W counter. **/
  int registerTensorRead(const Tensor & tensor,
                         VertexIdType node_id);
  /** Registers a new write on a Tensor. Returns the current epoch R/W counter. **/
  int registerTensorWrite(const Tensor & tensor,
                          VertexIdType node_id);

  /** Registers completion of an outstanding write on a Tensor.
      Returns the updated outstanding update count on the Tensor. **/
  std::size_t registerWriteCompletion(const Tensor & tensor);
  /** Returns the current outstanding update count on the tensor in the DAG. **/
  std::size_t getTensorUpdateCount(const Tensor & tensor);

  /** Registers a DAG node without dependencies. **/
  void registerDependencyFreeNode(VertexIdType node_id);
  /** Extracts a dependency-free node from the list.
      Returns FALSE if no such node exists. **/
  bool extractDependencyFreeNode(VertexIdType * node_id);

  /** Registers a DAG node as being executed (together with its execution handle). **/
  void registerExecutingNode(VertexIdType node_id,
                             TensorOpExecHandle exec_handle);
  /** Extracts an executed DAG node from the list. **/
  bool extractExecutingNode(VertexIdType * node_id);
  ExecutingNodesIterator extractExecutingNode(ExecutingNodesIterator node_iterator,
                                              VertexIdType * node_id);
  /** Returns a constant iterator to the list of currently executing DAG nodes. **/
  inline ExecutingNodesIterator executingNodesBegin() const {return nodes_executing_.cbegin();}
  inline ExecutingNodesIterator executingNodesEnd() const {return nodes_executing_.cend();}

  /** Moves the front node forward if the given DAG node is the next node
      after the front node and it has just been executed to completion. **/
  bool progressFrontNode(VertexIdType node_executed);

  /** Returns the front node id. **/
  VertexIdType getFrontNode() const;

private:
  /** Table for tracking the execution status of a given tensor:
      Tensor Hash --> TensorExecInfo **/
  std::unordered_map<TensorHashType,std::shared_ptr<TensorExecInfo>> tensor_info_;
  /** List of dependency-free unexecuted DAG nodes **/
  std::list<VertexIdType> nodes_ready_;
  /** List of the DAG nodes being currently executed **/
  std::list<std::pair<VertexIdType,TensorOpExecHandle>> nodes_executing_;
  /** Execution front node (all previous DAG nodes have been executed). **/
  VertexIdType front_node_;
};

} // namespace runtime
} // namespace exatn

#endif //EXATN_RUNTIME_TENSOR_EXEC_STATE_HPP_
