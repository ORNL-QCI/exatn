/** ExaTN:: Tensor Runtime: Directed acyclic graph of tensor operations
REVISION: 2019/07/19

Copyright (C) 2018-2019 Tiffany Mintz, Dmitry Lyakh, Alex McCaskey
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle)

Rationale:
 (a) Tensor graph is a directed acyclic graph in which vertices
     represent tensor operations and directed edges represent
     dependencies between them.

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

  // Adds a new vertex to the DAG (tensor operation)
  virtual VertexIdType addVertex(std::shared_ptr<numerics::TensorOperation> op) = 0;

  // Adds a directed edge between dependent and dependee vertices: dependent depends on dependee
  virtual void addEdge(VertexIdType dependent,
                       VertexIdType dependee) = 0;

  // Returns the properties (TensorOpNode) of a given vertex
  virtual const TensorOpNode & getVertexProperties(VertexIdType vertex_id) = 0;

  // Marks the vertex as executed
  virtual void setNodeExecuted(VertexIdType vertex_id) = 0;

  // Returns TRUE if the node has been executed to completion
  virtual bool nodeExecuted(VertexIdType vertex_id) = 0;

  // Returns TRUE if there is a dependency between vertices (directed edge)
  virtual bool edgeExists(VertexIdType vertex_id1,
                          VertexIdType vertex_id2) = 0;

  // Returns the number of vertices this vertex is connected to
  virtual std::size_t degree(VertexIdType vertex_id) = 0;

  // Returns the total number of edges
  virtual std::size_t size() = 0;

  // Returns the total number of vertices
  virtual std::size_t order() = 0;

  // Returns the list of vertices connected to the given vertex
  virtual std::vector<VertexIdType> getNeighborList(VertexIdType vertex_id) = 0;

  // Computes the shortest path from the start index
  virtual void computeShortestPath(VertexIdType startIndex,
                                   std::vector<double> & distances,
                                   std::vector<VertexIdType> & paths) = 0;

  // Clones (needed for plugin registry)
  virtual std::shared_ptr<TensorGraph> clone() = 0;

};

} // namespace runtime
} // namespace exatn

#endif //EXATN_RUNTIME_TENSOR_GRAPH_HPP_
