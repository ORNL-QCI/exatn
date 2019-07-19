#include "DirectedBoostGraph.hpp"

using namespace boost;

namespace exatn {
namespace runtime {

DirectedBoostGraph::DirectedBoostGraph() {
  graph_ = std::make_shared<d_adj_list>();
}

void DirectedBoostGraph::addEdge(const std::shared_ptr<TensorOpNode> &srcNode, const std::shared_ptr<TensorOpNode> &tgtNode) {
  add_edge(vertex(srcNode->id, *graph_.get()), vertex(tgtNode->id, *graph_.get()), *graph_.get());
}

/*
void DirectedBoostGraph::addVertex(TensorOpNode &&properties) {
  auto v = add_vertex(*graph_.get());
  (*graph_.get())[v].properties = properties;
  (*graph_.get())[v].properties.id=v;
}
*/

void DirectedBoostGraph::addVertex(std::shared_ptr<TensorOpNode> properties) {
  auto v = add_vertex(*graph_.get());
  (*graph_.get())[v].properties = properties;
  (*graph_.get())[v].properties->id=v;
}

const std::shared_ptr<TensorOpNode> &DirectedBoostGraph::getVertexProperties(const int index) {
  return (*graph_.get())[index].properties;
}

void DirectedBoostGraph::setNodeExecuted(const int index) {
  (*graph_.get())[index].properties->executed = true;
}

bool DirectedBoostGraph::nodeExecuted(const int index) {
  return (*graph_.get())[index].properties->executed;
}

bool DirectedBoostGraph::edgeExists(const int srcIndex, const int tgtIndex) {
  auto v1 = vertex(srcIndex, *graph_.get());
  auto v2 = vertex(tgtIndex, *graph_.get());
  auto p = edge(v1, v2, *graph_.get());
  return p.second;
}

int DirectedBoostGraph::degree(const int index) {
  return getNeighborList(index)
      .size(); // boost::degree(vertex(index, *graph_.get()), *graph_.get());
}

int DirectedBoostGraph::size() { return num_edges(*graph_.get()); }

int DirectedBoostGraph::order() { return num_vertices(*graph_.get()); }

std::vector<int> DirectedBoostGraph::getNeighborList(const int index) {
  std::vector<int> l;

  typedef typename boost::property_map<d_adj_list, boost::vertex_index_t>::type
      IndexMap;
  IndexMap indexMap = get(boost::vertex_index, *graph_.get());

  typedef typename boost::graph_traits<d_adj_list>::adjacency_iterator
      adjacency_iterator;

  std::pair<adjacency_iterator, adjacency_iterator> neighbors =
      boost::adjacent_vertices(vertex(index, *graph_.get()), *graph_.get());

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
      get(edge_weight, *graph_.get());
  std::vector<d_vertex_type> p(num_vertices(*graph_.get()));
  std::vector<int> d(num_vertices(*graph_.get()));
  d_vertex_type s = vertex(startIndex, *graph_.get());

  dijkstra_shortest_paths(
      *graph_.get(), s,
      predecessor_map(boost::make_iterator_property_map(
                          p.begin(), get(boost::vertex_index, *graph_.get())))
          .distance_map(boost::make_iterator_property_map(
              d.begin(), get(boost::vertex_index, *graph_.get()))));

  for (auto &di : d)
    distances.push_back(di);
  for (auto &pi : p)
    paths.push_back(pi);
}

} // namespace runtime
} // namespace exatn
