/*******************************************************************************
 * Copyright (c) 2019 UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * and Eclipse Distribution License v1.0 which accompanies this
 * distribution. The Eclipse Public License is available at
 * http://www.eclipse.org/legal/epl-v10.html and the Eclipse Distribution
 *License is available at https://eclipse.org/org/documents/edl-v10.php
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *******************************************************************************/
#ifndef EXATN_RUNTIME_DAG_HPP_
#define EXATN_RUNTIME_DAG_HPP_

#include "TensorGraph.hpp"
#include "tensor_operation.hpp"

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
#include <memory>

using namespace boost;

namespace exatn {
namespace runtime {

class DirectedBoostVertex {
public:
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

protected:

  DirectedGraphType graph_;

public:

  DirectedBoostGraph();

  VertexIdType addOperation(std::shared_ptr<numerics::TensorOperation> op) override;

  void addDependency(VertexIdType dependent,
                     VertexIdType dependee) override;

  const TensorOpNode & getNodeProperties(VertexIdType vertex_id) override;

  void setNodeExecuted(VertexIdType vertex_id) override;

  bool nodeExecuted(VertexIdType vertex_id) override;

  bool dependencyExists(VertexIdType vertex_id1,
                        VertexIdType vertex_id2) override;

  std::size_t degree(VertexIdType vertex_id) override;

  std::size_t getNumDependencies() override;

  std::size_t getNumNodes() override;

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

};

} // namespace runtime
} // namespace exatn

#endif //EXATN_RUNTIME_DAG_HPP_
