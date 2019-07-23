/** ExaTN:: Tensor Runtime: Directed acyclic graph of tensor operations
REVISION: 2019/07/22

Copyright (C) 2018-2019 Tiffany Mintz, Dmitry Lyakh, Alex McCaskey
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle)

Rationale:
 (a) Tensor graph is a directed acyclic graph in which vertices
     represent tensor operations and directed edges represent
     dependencies between them: A directed edge from node1 to
     node2 indicates that node1 depends on node2. Each DAG node
     has its unique integer id returned when the node is added to DAG.
 (b) Each tensor graph has its execution state.
**/

#ifndef EXATN_RUNTIME_TENSOR_GRAPH_HPP_
#define EXATN_RUNTIME_TENSOR_GRAPH_HPP_

#include "Identifiable.hpp"
#include "tensor_operation.hpp"
#include "tensor.hpp"

#include <iostream>
#include <unordered_map>
#include <deque>
#include <vector>
#include <string>
#include <memory>
#include <mutex>

namespace exatn {
namespace runtime {

// Graph node id:
using VertexIdType = std::size_t; //must match with boost::graph vertex descriptor type

// Tensor hash:
using numerics::TensorHashType;


// Tensor graph node
class TensorOpNode {

public:

  TensorOpNode():
   op_(nullptr), is_noop_(true), executing_(false), executed_(false) {}

  TensorOpNode(std::shared_ptr<numerics::TensorOperation> tens_op):
   op_(tens_op), is_noop_(false), executing_(false), executed_(false) {}

  TensorOpNode(const TensorOpNode &) = default;
  TensorOpNode & operator=(const TensorOpNode &) = default;
  TensorOpNode(TensorOpNode &&) noexcept = default;
  TensorOpNode & operator=(TensorOpNode &&) noexcept = default;
  ~TensorOpNode() = default;

  inline std::shared_ptr<numerics::TensorOperation> & getOperation() {return op_;}
  inline VertexIdType getId() const {return id_;}
  inline bool isDummy() const {return is_noop_;}
  inline bool isExecuting() const {return executing_;}
  inline bool isExecuted() const {return executed_;}

  inline void setId(VertexIdType id) {
    id_ = id; return;
  }

  inline void setExecuting() {
    assert(executing_ == false && executed_ == false);
    executing_ = true;
    return;
  }

  inline void setExecuted() {
    assert(executing_ == true && executed_ == false);
    executed_ = true; executing_ = false;
    return;
  }

  inline void lock() {mtx_.lock();}
  inline void unlock() {mtx_.unlock();}

protected:

  std::shared_ptr<numerics::TensorOperation> op_; //stored tensor operation
  bool is_noop_;    //TRUE if the stored tensor operation is NOOP (dummy node)
  bool executing_;  //TRUE if the stored tensor operation is currently being executed
  bool executed_;   //TRUE if the stored tensor operation has been executed to completion
  VertexIdType id_; //vertex id
  std::mutex mtx_;  //mutex

};


// Public Tensor Graph API
class TensorGraph : public Identifiable, public Cloneable<TensorGraph> {

public:

  /** Adds a new node (tensor operation) to the DAG and returns its id **/
  virtual VertexIdType addOperation(std::shared_ptr<numerics::TensorOperation> op) = 0;

  /** Adds a directed edge between dependent and dependee DAG nodes:
      dependent depends on dependee **/
  virtual void addDependency(VertexIdType dependent,
                             VertexIdType dependee) = 0;

  /** Returns the properties (TensorOpNode) of a given DAG node **/
  virtual TensorOpNode & getNodeProperties(VertexIdType vertex_id) = 0;

  /** Marks the DAG node as being executed **/
  virtual void setNodeExecuting(VertexIdType vertex_id) = 0;

  /** Marks the DAG node as executed to completion **/
  virtual void setNodeExecuted(VertexIdType vertex_id) = 0;

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

  // Clones (needed for plugin registry)
  virtual std::shared_ptr<TensorGraph> clone() = 0;

  inline void lock() {mtx_.lock();}
  inline void unlock() {mtx_.unlock();}

protected:

  /** Table for tracking the execution status on a given tensor:
      Tensor Hash --> Number of outstanding update operations **/
  std::unordered_map<TensorHashType,int> tensor_update_cnt_;
  /** Table for tracking last read or write access on a given tensor:
      Tensor Hash --> Last node of DAG performing read or write on this tensor **/
  std::unordered_map<TensorHashType,VertexIdType> tensor_last_read_;
  std::unordered_map<TensorHashType,VertexIdType> tensor_last_write_;
  /** List of (some) dependency-free DAG nodes **/
  std::deque<VertexIdType> nodes_ready_;
  /** Access mutex **/
  std::mutex mtx_;

};

} // namespace runtime
} // namespace exatn

#endif //EXATN_RUNTIME_TENSOR_GRAPH_HPP_
