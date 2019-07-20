#include "DirectedBoostGraph.hpp"

using namespace boost;

namespace exatn {
namespace runtime {

DirectedBoostGraph::DirectedBoostGraph() {
  graph_ = std::make_shared<d_adj_list>();
}


VertexIdType DirectedBoostGraph::addOperation(std::shared_ptr<numerics::TensorOperation> op) {
  auto vid = add_vertex(*graph_.get());
  (*graph_.get())[vid].properties = std::move(std::make_shared<TensorOpNode>(op));
  (*graph_.get())[vid].properties->id = vid;
  return vid; //new node id in the DAG
}


void DirectedBoostGraph::addDependency(VertexIdType dependent, VertexIdType dependee) {
  add_edge(vertex(dependent,*graph_.get()), vertex(dependee,*graph_.get()), *graph_.get());
  return;
}


const TensorOpNode & DirectedBoostGraph::getNodeProperties(VertexIdType vertex_id) {
  return *((*graph_.get())[vertex_id].properties.get());
}


void DirectedBoostGraph::setNodeExecuted(VertexIdType vertex_id) {
  (*graph_.get())[vertex_id].properties->executed = true;
  return;
}


bool DirectedBoostGraph::nodeExecuted(VertexIdType vertex_id) {
  return (*graph_.get())[vertex_id].properties->executed;
}


bool DirectedBoostGraph::dependencyExists(VertexIdType vertex_id1, VertexIdType vertex_id2) {
  auto vid1 = vertex(vertex_id1, *graph_.get());
  auto vid2 = vertex(vertex_id2, *graph_.get());
  auto p = edge(vid1, vid2, *graph_.get());
  return p.second;
}


std::size_t DirectedBoostGraph::degree(VertexIdType vertex_id) {
  return getNeighborList(vertex_id).size(); // boost::degree(vertex(index, *graph_.get()), *graph_.get());
}


std::size_t DirectedBoostGraph::getNumDependencies() {
  return num_edges(*graph_.get());
}


std::size_t DirectedBoostGraph::getNumNodes() {
  return num_vertices(*graph_.get());
}


std::vector<VertexIdType> DirectedBoostGraph::getNeighborList(VertexIdType vertex_id) {
  std::vector<VertexIdType> l;

  typedef typename boost::property_map<d_adj_list, boost::vertex_index_t>::type IndexMap;
  IndexMap indexMap = get(boost::vertex_index, *graph_.get());

  typedef typename boost::graph_traits<d_adj_list>::adjacency_iterator adjacency_iterator;

  std::pair<adjacency_iterator, adjacency_iterator> neighbors =
    boost::adjacent_vertices(vertex(vertex_id,*graph_.get()), *graph_.get());

  for (; neighbors.first != neighbors.second; ++neighbors.first) {
    VertexIdType neighborIdx = indexMap[*neighbors.first];
    l.push_back(neighborIdx);
  }

  return l;
}


void DirectedBoostGraph::computeShortestPath(VertexIdType startIndex,
                                             std::vector<double> & distances,
                                             std::vector<VertexIdType> & paths) {
  typename property_map<d_adj_list, edge_weight_t>::type weightmap =
           get(edge_weight, *graph_.get());
  std::vector<VertexIdType> p(num_vertices(*graph_.get()));
  std::vector<std::size_t> d(num_vertices(*graph_.get()));
  d_vertex_type s = vertex(startIndex, *graph_.get());

  dijkstra_shortest_paths(
      *graph_.get(), s,
      predecessor_map(boost::make_iterator_property_map(
                             p.begin(), get(boost::vertex_index, *graph_.get())))
          .distance_map(boost::make_iterator_property_map(
                               d.begin(), get(boost::vertex_index, *graph_.get()))));

  for (const auto & di: d) distances.push_back(static_cast<double>(di));
  for (const auto & pi: p) paths.push_back(pi);
  return;
}

} // namespace runtime
} // namespace exatn
