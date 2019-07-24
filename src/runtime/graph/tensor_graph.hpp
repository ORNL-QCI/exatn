/** ExaTN:: Tensor Runtime: Directed acyclic graph of tensor operations
REVISION: 2019/07/23

Copyright (C) 2018-2019 Tiffany Mintz, Dmitry Lyakh, Alex McCaskey
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle)

Rationale:
 (a) Tensor graph is a directed acyclic graph in which vertices
     represent tensor operations and directed edges represent
     dependencies between them: A directed edge from node1 to
     node2 indicates that node1 depends on node2. Each DAG node
     has its unique integer vertex id (VertexIdType) returned
     when the node is added to the DAG.
 (b) The tensor graph contains:
     1. The DAG implementation (in the directed Boost graph subclass);
     2. The DAG execution state (TensorExecState data member).
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
  inline bool isExecuting() const {return executing_;}
  inline bool isExecuted() const {return executed_;}

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
  std::recursive_mutex mtx_; //mutex
};


// Public Tensor Graph API
class TensorGraph : public Identifiable, public Cloneable<TensorGraph> {

public:
  TensorGraph() = default;
  TensorGraph(const TensorGraph &) = delete;
  TensorGraph & operator=(const TensorGraph &) = delete;
  TensorGraph(TensorGraph &&) noexcept = default;
  TensorGraph & operator=(TensorGraph &&) noexcept = default;
  ~TensorGraph() = default;

  /** Adds a new node (tensor operation) to the DAG and returns its id **/
  virtual VertexIdType addOperation(std::shared_ptr<TensorOperation> op) = 0;

  /** Adds a directed edge between dependent and dependee DAG nodes:
      dependent depends on dependee (dependent --> dependee) **/
  virtual void addDependency(VertexIdType dependent,
                             VertexIdType dependee) = 0;

  /** Returns the properties (TensorOpNode) of a given DAG node **/
  virtual TensorOpNode & getNodeProperties(VertexIdType vertex_id) = 0;

  /** Marks the DAG node as being executed **/
  virtual void setNodeExecuting(VertexIdType vertex_id) = 0;

  /** Marks the DAG node as executed to completion **/
  virtual void setNodeExecuted(VertexIdType vertex_id,
                               int error_code = 0) = 0;

  /** Returns TRUE if the DAG node is currently being executed **/
  virtual bool nodeExecuting(VertexIdType vertex_id) = 0;

  /** Returns TRUE if the DAG node has been executed to completion **/
  virtual bool nodeExecuted(VertexIdType vertex_id) = 0;

  /** Returns TRUE if there is a dependency between two DAG nodes:
      If vertex_id1 node depends on vertex_id2 node **/
  virtual bool dependencyExists(VertexIdType vertex_id1,
                                VertexIdType vertex_id2) = 0;

  /** Returns the number of nodes the given node is connected to **/
  virtual std::size_t degree(VertexIdType vertex_id) = 0;

  /** Returns the total number of dependencies (directed edges) in the DAG **/
  virtual std::size_t getNumDependencies() = 0;

  /** Returns the total number of nodes in the DAG **/
  virtual std::size_t getNumNodes() = 0;

  /** Returns the list of nodes connected to the given DAG node **/
  virtual std::vector<VertexIdType> getNeighborList(VertexIdType vertex_id) = 0;

  /** Computes the shortest path from the start index **/
  virtual void computeShortestPath(VertexIdType startIndex,
                                   std::vector<double> & distances,
                                   std::vector<VertexIdType> & paths) = 0;

  // Clones an empty subclass instance (needed for plugin registry)
  virtual std::shared_ptr<TensorGraph> clone() = 0;

protected:
  TensorExecState exec_state_; //tensor graph execution state
};

} // namespace runtime
} // namespace exatn

#endif //EXATN_RUNTIME_TENSOR_GRAPH_HPP_
