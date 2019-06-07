
#include "DirectedBoostGraph.hpp"

#include <numeric>
#include <regex>
#include <utility>

using namespace boost;

namespace exatn {

DirectedBoostGraph::DirectedBoostGraph() {
  _graph = std::make_shared<d_adj_list>();
}

void DirectedBoostGraph::addEdge(const std::shared_ptr<TensorOpNode> &srcNode, const std::shared_ptr<TensorOpNode> &tgtNode) {
  add_edge(vertex(srcNode.id, *_graph.get()), vertex(tgtNode.id, *_graph.get()),
           *_graph.get());
}

/*
void DirectedBoostGraph::addVertex(TensorOpNode &&properties) {
  auto v = add_vertex(*_graph.get());
  (*_graph.get())[v].properties = properties;
  (*_graph.get())[v].properties.id=v;
}
*/

void DirectedBoostGraph::addVertex(std::shared_ptr<TensorOpNode> properties) {
  auto v = add_vertex(*_graph.get());
  (*_graph.get())[v].properties = properties;
  (*_graph.get())[v].properties->id=v;
}

const std::shared_ptr<TensorOpNode> &DirectedBoostGraph::getVertexProperties(const int index) {
  return (*_graph.get())[index].properties;
}

void DirectedBoostGraph::setNodeExecuted(const int index) {
  (*_graph.get())[index].properties->executed = true;
}

bool DirectedBoostGraph::edgeExists(const int srcIndex, const int tgtIndex) {
  auto v1 = vertex(srcIndex, *_graph.get());
  auto v2 = vertex(tgtIndex, *_graph.get());
  auto p = edge(v1, v2, *_graph.get());
  return p.second;
}

int DirectedBoostGraph::degree(const int index) {
  return getNeighborList(index)
      .size(); // boost::degree(vertex(index, *_graph.get()), *_graph.get());
}

int DirectedBoostGraph::size() { return num_edges(*_graph.get()); }

int DirectedBoostGraph::order() { return num_vertices(*_graph.get()); }

std::vector<int> DirectedBoostGraph::getNeighborList(const int index) {
  std::vector<int> l;

  typedef typename boost::property_map<d_adj_list, boost::vertex_index_t>::type
      IndexMap;
  IndexMap indexMap = get(boost::vertex_index, *_graph.get());

  typedef typename boost::graph_traits<d_adj_list>::adjacency_iterator
      adjacency_iterator;

  std::pair<adjacency_iterator, adjacency_iterator> neighbors =
      boost::adjacent_vertices(vertex(index, *_graph.get()), *_graph.get());

  for (; neighbors.first != neighbors.second; ++neighbors.first) {
    int neighborIdx = indexMap[*neighbors.first];
    l.push_back(neighborIdx);
  }

  return l;
}

void DirectedBoostGraph::computeShortestPath(int startIndex,
                                             std::vector<double> &distances,
                                             std::vector<int> &paths) {
  typename property_map<d_adj_list, edge_weight_t>::type weightmap =
      get(edge_weight, *_graph.get());
  std::vector<d_vertex_type> p(num_vertices(*_graph.get()));
  std::vector<int> d(num_vertices(*_graph.get()));
  d_vertex_type s = vertex(startIndex, *_graph.get());

  dijkstra_shortest_paths(
      *_graph.get(), s,
      predecessor_map(boost::make_iterator_property_map(
                          p.begin(), get(boost::vertex_index, *_graph.get())))
          .distance_map(boost::make_iterator_property_map(
              d.begin(), get(boost::vertex_index, *_graph.get()))));

  for (auto &di : d)
    distances.push_back(di);
  for (auto &pi : p)
    paths.push_back(pi);
}

} // namespace exatn
