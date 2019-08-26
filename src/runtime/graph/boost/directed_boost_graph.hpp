/** ExaTN:: Tensor Runtime: Directed acyclic graph of tensor operations
REVISION: 2019/08/26

Copyright (C) 2018-2019 Tiffany Mintz, Dmitry Lyakh, Alex McCaskey
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle)

Rationale:
 (a) Tensor graph is a directed acyclic graph in which vertices
     represent tensor operations and directed edges represent
     dependencies between them: A directed edge from node1 to
     node2 indicates that node1 depends on node2. Each DAG node
     has its unique integer vertex id (VertexIdType) returned
     when the node is appended to the DAG.
 (b) The tensor graph contains:
     1. The DAG implementation (DirectedBoostGraph subclass);
     2. The DAG execution state (TensorExecState data member).
**/

#ifndef EXATN_RUNTIME_DAG_HPP_
#define EXATN_RUNTIME_DAG_HPP_

#include "tensor_graph.hpp"
#include "tensor_operation.hpp"
#include "tensor.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dag_shortest_paths.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/eccentricity.hpp>
#include <boost/graph/exterior_property.hpp>
#include <boost/graph/floyd_warshall_shortest.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/property_map/property_map.hpp>

#include <type_traits>
#include <string>
#include <memory>

using namespace boost;

namespace exatn {
namespace runtime {

struct DirectedBoostVertex {
  std::shared_ptr<TensorOpNode> properties; //properties of the DAG node
};


using d_adj_list = adjacency_list<vecS, vecS, directedS, DirectedBoostVertex,
      boost::property<boost::edge_weight_t, double>>;

using DirectedGraphType = std::shared_ptr<d_adj_list>;

using d_vertex_type = typename boost::graph_traits<adjacency_list<
      vecS, vecS, directedS, DirectedBoostVertex,
      boost::property<boost::edge_weight_t, double>>>::vertex_descriptor;

using d_edge_type = typename boost::graph_traits<adjacency_list<
      vecS, vecS, directedS, DirectedBoostVertex,
      boost::property<boost::edge_weight_t, double>>>::edge_descriptor;

static_assert(std::is_same<d_vertex_type,VertexIdType>::value,"Vertex id type mismatch!");


class DirectedBoostGraph : public TensorGraph {

public:
  DirectedBoostGraph();
  DirectedBoostGraph(const DirectedBoostGraph &) = delete;
  DirectedBoostGraph & operator=(const DirectedBoostGraph &) = delete;
  DirectedBoostGraph(DirectedBoostGraph &&) noexcept = default;
  DirectedBoostGraph & operator=(DirectedBoostGraph &&) noexcept = default;
  virtual ~DirectedBoostGraph() = default;

  VertexIdType addOperation(std::shared_ptr<TensorOperation> op) override;

  void addDependency(VertexIdType dependent,
                     VertexIdType dependee) override;

  bool dependencyExists(VertexIdType vertex_id1,
                        VertexIdType vertex_id2) override;

  TensorOpNode & getNodeProperties(VertexIdType vertex_id) override;

  std::size_t getNodeDegree(VertexIdType vertex_id) override;

  std::size_t getNumNodes() override;

  std::size_t getNumDependencies() override;

  std::vector<VertexIdType> getNeighborList(VertexIdType vertex_id) override;

  void computeShortestPath(VertexIdType startIndex,
                           std::vector<double> & distances,
                           std::vector<VertexIdType> & paths) override;

  const std::string name() const override {
    return "boost-digraph";
  }

  const std::string description() const override {
    return "Directed acyclic graph of tensor operations";
  }

  std::shared_ptr<TensorGraph> clone() override {
    return std::make_shared<DirectedBoostGraph>();
  }

protected:
  DirectedGraphType dag_; //std::shared_ptr<d_adj_list>
};

} // namespace runtime
} // namespace exatn

#endif //EXATN_RUNTIME_DAG_HPP_
