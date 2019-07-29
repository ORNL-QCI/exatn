/** ExaTN:: Tensor Runtime: Directed acyclic graph (DAG) of tensor operations
REVISION: 2019/07/29

Copyright (C) 2018-2019 Tiffany Mintz, Dmitry Lyakh, Alex McCaskey
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle)

Rationale:
 (a) The execution space consists of one or more DAGs in which nodes
     represent tensor operations (tasks) and directed edges represent
     dependencies between the corresponding nodes (tensor operations).
     Each DAG is associated with a uniquely named TAProL scope such that
     all tensor operations submitted by the Client to the ExaTN numerics
     server are forwarded into the DAG associated with the TaProL scope
     in which the Client currently resides.
 (b) The tensor graph contains:
     1. The DAG implementation (in the directed Boost graph subclass);
     2. The DAG execution state (TensorExecState data member).
 (c) DEVELOPERS ONLY: The TensorGraph object provides lock/unlock methods for concurrent update
     of the DAG structure (by Client thread) and its execution state (by Execution thread).
     Additionally each node of the TensorGraph (TensorOpNode object) provides more fine grain
     locking mechanism (lock/unlock methods) for providing exclusive access to individual DAG nodes.
**/

#ifndef EXATN_RUNTIME_TENSOR_GRAPH_HPP_
#define EXATN_RUNTIME_TENSOR_GRAPH_HPP_

#include "Identifiable.hpp"

#include "tensor_exec_state.hpp"
#include "tensor_operation.hpp"
#include "tensor.hpp"

#include <vector>
#include <memory>
#include <mutex>

namespace exatn {
namespace runtime {

// Tensor Graph node
class TensorOpNode {

public:
  TensorOpNode():
   op_(nullptr), is_noop_(true), executing_(false), executed_(false), error_(0)
  {}

  TensorOpNode(std::shared_ptr<TensorOperation> tens_op):
   op_(tens_op), is_noop_(false), executing_(false), executed_(false), error_(0)
  {}

  TensorOpNode(const TensorOpNode &) = delete;
  TensorOpNode & operator=(const TensorOpNode &) = delete;
  TensorOpNode(TensorOpNode &&) noexcept = default;
  TensorOpNode & operator=(TensorOpNode &&) noexcept = default;
  ~TensorOpNode() = default;

  inline std::shared_ptr<TensorOperation> & getOperation() {return op_;}
  inline VertexIdType getId() const {return id_;}
  inline bool isDummy() const {return is_noop_;}
  inline bool isExecuting() {return executing_;}
  inline bool isExecuted(int * error_code = nullptr) {
    if(error_code != nullptr) *error_code = error_;
    return executed_;
  }

  inline void setId(VertexIdType id) {
    id_ = id;
    return;
  }

  inline void setExecuting() {
    assert(executing_ == false && executed_ == false);
    executing_ = true;
    return;
  }

  inline void setExecuted(int error_code = 0) {
    assert(executing_ == true && executed_ == false);
    executed_ = true; executing_ = false;
    error_ = error_code;
    return;
  }

  inline void lock() {mtx_.lock();}
  inline void unlock() {mtx_.unlock();}

protected:
  std::shared_ptr<TensorOperation> op_; //stored tensor operation
  bool is_noop_;    //TRUE if the stored tensor operation is NOOP (dummy node)
  bool executing_;  //TRUE if the stored tensor operation is currently being executed
  bool executed_;   //TRUE if the stored tensor operation has been executed to completion
  int error_;       //execution error code (0:success)
  VertexIdType id_; //vertex id
  std::recursive_mutex mtx_; //object access mutex
};


// Public Tensor Graph API
class TensorGraph : public Identifiable, public Cloneable<TensorGraph> {

public:
  TensorGraph() = default;
  TensorGraph(const TensorGraph &) = delete;
  TensorGraph & operator=(const TensorGraph &) = delete;
  TensorGraph(TensorGraph &&) noexcept = default;
  TensorGraph & operator=(TensorGraph &&) noexcept = default;
  virtual ~TensorGraph() = default;

  /** Adds a new node (tensor operation) to the DAG and returns its id **/
  virtual VertexIdType addOperation(std::shared_ptr<TensorOperation> op) = 0;

  /** Adds a directed edge between dependent and dependee DAG nodes:
      <dependent> depends on <dependee> (dependent --> dependee) **/
  virtual void addDependency(VertexIdType dependent,
                             VertexIdType dependee) = 0;

  /** Returns TRUE if there is a dependency between two DAG nodes:
      If vertex_id1 node depends on vertex_id2 node **/
  virtual bool dependencyExists(VertexIdType vertex_id1,
                                VertexIdType vertex_id2) = 0;

  /** Returns the properties (TensorOpNode) of a given DAG node **/
  virtual TensorOpNode & getNodeProperties(VertexIdType vertex_id) = 0;

  /** Returns the number of nodes the given node is connected to **/
  virtual std::size_t getNodeDegree(VertexIdType vertex_id) = 0;

  /** Returns the total number of nodes in the DAG **/
  virtual std::size_t getNumNodes() = 0;

  /** Returns the total number of dependencies (directed edges) in the DAG **/
  virtual std::size_t getNumDependencies() = 0;

  /** Returns the list of nodes connected to the given DAG node **/
  virtual std::vector<VertexIdType> getNeighborList(VertexIdType vertex_id) = 0;

  /** Computes the shortest path from the start index **/
  virtual void computeShortestPath(VertexIdType startIndex,
                                   std::vector<double> & distances,
                                   std::vector<VertexIdType> & paths) = 0;

  /** Clones an empty subclass instance (needed for plugin registry) **/
  virtual std::shared_ptr<TensorGraph> clone() = 0;

  /** Marks the DAG node as being executed **/
  void setNodeExecuting(VertexIdType vertex_id) {
    return getNodeProperties(vertex_id).setExecuting();
  }

  /** Marks the DAG node as executed to completion **/
  void setNodeExecuted(VertexIdType vertex_id, int error_code = 0) {
    return getNodeProperties(vertex_id).setExecuted(error_code);
  }

  /** Returns TRUE if the DAG node is currently being executed **/
  bool nodeExecuting(VertexIdType vertex_id) {
    return getNodeProperties(vertex_id).isExecuting();
  }

  /** Returns TRUE if the DAG node has been executed to completion,
      error_code will return the error code if executed. **/
  bool nodeExecuted(VertexIdType vertex_id, int * error_code = nullptr) {
    return getNodeProperties(vertex_id).isExecuted(error_code);
  }

  /** Returns the current outstanding update count on the tensor in the DAG. **/
  inline std::size_t getTensorUpdateCount(const Tensor & tensor) {
    return exec_state_.getTensorUpdateCount(tensor);
  }

  /** Registers a DAG node without dependencies. **/
  inline void registerDependencyFreeNode(VertexIdType node_id) {
    return exec_state_.registerDependencyFreeNode(node_id);
  }

  /** Extracts a dependency-free node from the list.
      Returns FALSE if no such node exists. **/
  inline bool extractDependencyFreeNode(VertexIdType * node_id) {
    return exec_state_.extractDependencyFreeNode(node_id);
  }

  /** Registers a DAG node as being executed. **/
  inline void registerExecutingNode(VertexIdType node_id) {
    return exec_state_.registerExecutingNode(node_id);
  }

  /** Extracts an executed DAG node from the list of executing nodes. **/
  inline bool extractExecutingNode(VertexIdType * node_id) {
    return exec_state_.extractExecutingNode(node_id);
  }

  inline void lock() {mtx_.lock();}
  inline void unlock() {mtx_.unlock();}

protected:
  TensorExecState exec_state_; //tensor graph execution state
  std::recursive_mutex mtx_; //object access mutex
};

} // namespace runtime
} // namespace exatn

#endif //EXATN_RUNTIME_TENSOR_GRAPH_HPP_
