/** ExaTN:: Tensor Runtime: Tensor graph execution state
REVISION: 2019/07/25

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

public:
  TensorExecState() = default;
  TensorExecState(const TensorExecState &) = delete;
  TensorExecState & operator=(const TensorExecState &) = delete;
  TensorExecState(TensorExecState &&) noexcept = default;
  TensorExecState & operator=(TensorExecState &&) noexcept = default;
  ~TensorExecState() = default;

  /** Registers an update operation on a tensor in the DAG.
      Returns the updated outstanding update count on the tensor. **/
  int incrTensorUpdate(const Tensor & tensor);
  /** Registers completion of an update operation on a tensor in the DAG.
      Returns the updated outstanding update count on the tensor. **/
  int decrTensorUpdate(const Tensor & tensor);

  /** Updates the last DAG node id performing a read on a given tensor. **/
  void updateLastTensorRead(const Tensor & tensor, VertexIdType node_id);
  /** Updates the last DAG node id performing a write on a given tensor. **/
  void updateLastTensorWrite(const Tensor & tensor, VertexIdType node_id);
  /** Clears the last read on a tensor **/
  void clearLastTensorRead(const Tensor & tensor);
  /** Clears the last write on a tensor. **/
  void clearLastTensorWrite(const Tensor & tensor);

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
  /** Table for tracking the execution status on a given tensor:
      Tensor Hash --> Number of outstanding update operations on the Tensor **/
  std::unordered_map<TensorHashType,int> tensor_update_cnt_;
  /** Table for tracking last read or write access on a given tensor:
      Tensor Hash --> Last node of DAG performing read or write on the Tensor **/
  std::unordered_map<TensorHashType,VertexIdType> tensor_last_read_;
  std::unordered_map<TensorHashType,VertexIdType> tensor_last_write_;
  /** List of dependency-free unexecuted DAG nodes **/
  std::list<VertexIdType> nodes_ready_;
  /** List of the currently executed DAG nodes **/
  std::list<VertexIdType> nodes_executing_;
};

} // namespace runtime
} // namespace exatn

#endif //EXATN_RUNTIME_TENSOR_EXEC_STATE_HPP_
