/** ExaTN:: Tensor Runtime: Directed acyclic graph of tensor operations
REVISION: 2019/07/18

Copyright (C) 2018-2019 Tiffany Mintz, Dmitry Lyakh, Alex McCaskey
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle)

Rationale:
 (a) Tensor graph is a directed acyclic graph in which vertices
     represent tensor operations and directed edges represent
     dependencies between them.

**/

#ifndef EXATN_RUNTIME_GRAPH_HPP_
#define EXATN_RUNTIME_GRAPH_HPP_

#include "Identifiable.hpp"
#include "tensor_operation.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace exatn {
//namespace runtime {

// Tensor graph node
struct TensorOpNode {

  TensorOpNode() : op(nullptr) {}
  TensorOpNode(std::shared_ptr<numerics::TensorOperation> tens_op) : op(tens_op) {}

  std::shared_ptr<numerics::TensorOperation> op; //stored tensor operation
  bool executed = false; //execution status of the tensor operation
  bool is_noop = false; //
  int id; //
  // Add any other info you need
};

// Public Graph API
class TensorGraph : public Identifiable, public Cloneable<TensorGraph> {
public:
  // Add an edge between src and tgt, this is a directed edge
  virtual void addEdge(const std::shared_ptr<TensorOpNode> &srcNode, const std::shared_ptr<TensorOpNode> &tgtNode) = 0;

  virtual void addVertex(std::shared_ptr<TensorOpNode> opNode) = 0;
  //virtual void addVertex(TensorOpNode &&opNode) = 0;

  // For now lets assume as you build it,
  // you can't change the structure or the node values
  // virtual void removeEdge(const TensorOpNode &srcNode, const TensorOpNode &tgtNode) = 0;
  // virtual void setVertexProperties(const int index, TensorOpNode& opNode) =
  // 0; virtual void setVertexProperties(const int index, TensorOpNode&& opNode)
  // = 0;

  // Get the TensorOpNode at the given index
  virtual const std::shared_ptr<TensorOpNode> &getVertexProperties(const int index) = 0;

  // Flip the bool on the TensorOpNode to indicate this
  // node has been executed
  virtual void setNodeExecuted(const int index) = 0;

  virtual bool DirectedBoostGraph::nodeExecuted(const int index) = 0;

  // Return true if edge exists
  virtual bool edgeExists(const int srcIndex, const int tgtIndex) = 0;

  // Get how many vertices this vertex is connected to
  virtual int degree(const int index) = 0;

  // Get all vertex indices this vertex is connected to
  virtual std::vector<int> getNeighborList(const int index) = 0;

  // n edges
  virtual int size() = 0;

  // n vertices
  virtual int order() = 0;

  // Compute shortest path from start index
  virtual void computeShortestPath(int startIndex,
                                   std::vector<double> &distances,
                                   std::vector<int> &paths) = 0;

  // needed for plugin registry
  virtual std::shared_ptr<TensorGraph> clone() = 0;
};

//} // namespace runtime
} // namespace exatn

#endif //EXATN_RUNTIME_GRAPH_HPP_
