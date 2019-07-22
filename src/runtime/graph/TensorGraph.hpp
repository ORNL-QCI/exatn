/** ExaTN:: Tensor Runtime: Directed acyclic graph of tensor operations
REVISION: 2019/07/20

Copyright (C) 2018-2019 Tiffany Mintz, Dmitry Lyakh, Alex McCaskey
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle)

Rationale:
 (a) Tensor graph is a directed acyclic graph in which vertices
     represent tensor operations and directed edges represent
     dependencies between them: A directed edge from node1 to
     node2 indicates that node1 depends on node2. Each DAG node
     has its unique integer id returned when the node is added to DAG.
**/

#ifndef EXATN_RUNTIME_TENSOR_GRAPH_HPP_
#define EXATN_RUNTIME_TENSOR_GRAPH_HPP_

#include "Identifiable.hpp"
#include "tensor_operation.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace exatn {
namespace runtime {

using VertexIdType = std::size_t; //must match with boost::graph vertex descriptor type


// Tensor graph node
class TensorOpNode {
public:
  TensorOpNode(): op(nullptr), is_noop(true), executed(false) {}
  TensorOpNode(std::shared_ptr<numerics::TensorOperation> tens_op):
   op(tens_op), is_noop(false), executed(false) {}
  TensorOpNode(const TensorOpNode &) = default;
  TensorOpNode & operator=(const TensorOpNode &) = default;
  TensorOpNode(TensorOpNode &&) noexcept = default;
  TensorOpNode & operator=(TensorOpNode &&) noexcept = default;
  ~TensorOpNode() = default;
  //Members:
  std::shared_ptr<numerics::TensorOperation> op; //stored tensor operation
  bool is_noop; //TRUE if the stored tensor operation is NOOP
  bool executed; //TRUE if the tensor operation has been completed
  std::size_t id; //vertex id
};


// Public Graph API
class TensorGraph : public Identifiable, public Cloneable<TensorGraph> {
public:

  /** Adds a new node (tensor operation) to the DAG and returns its id **/
  virtual VertexIdType addOperation(std::shared_ptr<numerics::TensorOperation> op) = 0;

  /** Adds a directed edge between dependent and dependee DAG nodes:
      dependent depends on dependee **/
  virtual void addDependency(VertexIdType dependent,
                             VertexIdType dependee) = 0;

  /** Returns the properties (TensorOpNode) of a given DAG node **/
  virtual const TensorOpNode & getNodeProperties(VertexIdType vertex_id) = 0;

  /** Marks the DAG node as executed to completion **/
  virtual void setNodeExecuted(VertexIdType vertex_id) = 0;

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

};

} // namespace runtime
} // namespace exatn

#endif //EXATN_RUNTIME_TENSOR_GRAPH_HPP_
