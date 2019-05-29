#ifndef XACC_QUANTUM_GRAPH_HPP_
#define XACC_QUANTUM_GRAPH_HPP_

#include "Identifiable.hpp"

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace exatn {

// For now
using TensorOp = int;

class TensorOpNode {
public:
  TensorOpNode() : op(nullptr) {}
  TensorOpNode(TensorOp *o) : op(o) {}
  TensorOp *op;
  bool executed = false;
  int id;
  // Add any other info you need
};

// Public Graph API
class TensorGraph : public Identifiable, public Cloneable<TensorGraph> {
public:
  // Add an edge between src and tgt, this is
  // a directed edge
  virtual void addEdge(const TensorOpNode &srcNode, const TensorOpNode &tgtNode) = 0;

  virtual void addVertex(TensorOpNode &opNode) = 0;
  virtual void addVertex(TensorOpNode &&opNode) = 0;

  // For now lets assume as you build it,
  // you can't change the structure or the node values
  // virtual void removeEdge(const TensorOpNode &srcNode, const TensorOpNode &tgtNode) = 0;
  // virtual void setVertexProperties(const int index, TensorOpNode& opNode) =
  // 0; virtual void setVertexProperties(const int index, TensorOpNode&& opNode)
  // = 0;

  // Get the TensorOpNode at the given index
  virtual TensorOpNode &getVertexProperties(const int index) = 0;

  // Flip the bool on the TensorOpNode to indicate this
  // node has been executed
  virtual void setNodeExecuted(const int index) = 0;

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

} // namespace exatn
#endif
