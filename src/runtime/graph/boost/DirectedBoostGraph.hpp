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
#ifndef XACC_UTILS_IGRAPH_HPP_
#define XACC_UTILS_IGRAPH_HPP_

#include "TensorGraph.hpp"
#include <memory>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dag_shortest_paths.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/eccentricity.hpp>
#include <boost/graph/exterior_property.hpp>
#include <boost/graph/floyd_warshall_shortest.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/property_map/property_map.hpp>

using namespace boost;
namespace exatn {

class DirectedBoostVertex {
public:
  TensorOpNode properties;
};

using d_adj_list =
    adjacency_list<vecS, vecS, directedS, DirectedBoostVertex,
                   boost::property<boost::edge_weight_t, double>>;

using DirectedGraphType = std::shared_ptr<d_adj_list>;

using d_vertex_type = typename boost::graph_traits<adjacency_list<
    vecS, vecS, directedS, DirectedBoostVertex,
    boost::property<boost::edge_weight_t, double>>>::vertex_descriptor;

using d_edge_type = typename boost::graph_traits<adjacency_list<
    vecS, vecS, directedS, DirectedBoostVertex,
    boost::property<boost::edge_weight_t, double>>>::edge_descriptor;

class DirectedBoostGraph : public TensorGraph {

protected:
  DirectedGraphType _graph;

public:
  DirectedBoostGraph();

  void addEdge(const int srcIndex, const int tgtIndex) override;

  void addVertex(TensorOpNode &&properties) override;
  void addVertex(TensorOpNode &properties) override;

  TensorOpNode &getVertexProperties(const int index) override;
  void setNodeExecuted(const int index) override;

  bool edgeExists(const int srcIndex, const int tgtIndex) override;

  int degree(const int index) override;
  int size() override;
  int order() override;

  std::vector<int> getNeighborList(const int index) override;

  void computeShortestPath(int startIndex, std::vector<double> &distances,
                           std::vector<int> &paths) override;

  const std::string name() const override { return "boost-digraph"; }
  const std::string description() const override { return ""; }

  std::shared_ptr<TensorGraph> clone() override {
    return std::make_shared<DirectedBoostGraph>();
  }
};

} // namespace exatn

#endif
