#include "directed_boost_graph.hpp"

using namespace boost;

namespace exatn {
namespace runtime {

DirectedBoostGraph::DirectedBoostGraph() {
  dag_ = std::make_shared<d_adj_list>();
}


VertexIdType DirectedBoostGraph::addOperation(std::shared_ptr<TensorOperation> op) {
  this->lock();
  auto vid = add_vertex(*dag_);
  (*dag_)[vid].properties = std::move(std::make_shared<TensorOpNode>(op));
  (*dag_)[vid].properties->setId(vid); //DAG node id is stored in the node properties
  auto output_tensor = op->getTensorOperand(0); //output tensor operand
  if(output_tensor) exec_state_.incrTensorUpdate(*output_tensor);
  VertexIdType node_id;
  bool dependent = false;
  if(exec_state_.getLastTensorRead(*output_tensor,&node_id)){
    addDependency(vid,node_id); dependent = true;
  }
  if(exec_state_.getLastTensorWrite(*output_tensor,&node_id)){
    addDependency(vid,node_id); dependent = true;
  }
  exec_state_.updateLastTensorWrite(*output_tensor,vid);
  unsigned int num_operands = op->getNumOperands();
  for(unsigned int i = 1; i < num_operands; ++i){ //input tensor operands
    auto tensor = op->getTensorOperand(i);
    if(exec_state_.getLastTensorWrite(*tensor,&node_id)){
      addDependency(vid,node_id); dependent = true;
    }
    exec_state_.updateLastTensorRead(*tensor,vid);
  }
  if(!dependent) exec_state_.registerDependencyFreeNode(vid);
  this->unlock();
  return vid; //new node id in the DAG
}


void DirectedBoostGraph::addDependency(VertexIdType dependent, VertexIdType dependee) {
  add_edge(vertex(dependent,*dag_), vertex(dependee,*dag_), *dag_);
  return;
}


TensorOpNode & DirectedBoostGraph::getNodeProperties(VertexIdType vertex_id) {
  return *((*dag_)[vertex_id].properties);
}


void DirectedBoostGraph::setNodeExecuting(VertexIdType vertex_id) {
  (*dag_)[vertex_id].properties->setExecuting();
  return;
}


void DirectedBoostGraph::setNodeExecuted(VertexIdType vertex_id, int error_code) {
  (*dag_)[vertex_id].properties->setExecuted(error_code);
  return;
}


bool DirectedBoostGraph::nodeExecuting(VertexIdType vertex_id) {
  return (*dag_)[vertex_id].properties->isExecuting();
}


bool DirectedBoostGraph::nodeExecuted(VertexIdType vertex_id) {
  return (*dag_)[vertex_id].properties->isExecuted();
}


bool DirectedBoostGraph::dependencyExists(VertexIdType vertex_id1, VertexIdType vertex_id2) {
  auto vid1 = vertex(vertex_id1, *dag_);
  auto vid2 = vertex(vertex_id2, *dag_);
  auto p = edge(vid1, vid2, *dag_);
  return p.second;
}


std::size_t DirectedBoostGraph::getNodeDegree(VertexIdType vertex_id) {
//return boost::degree(vertex(vertex_id, *dag_), *dag_);
  return getNeighborList(vertex_id).size();
}


std::size_t DirectedBoostGraph::getNumDependencies() {
  return num_edges(*dag_);
}


std::size_t DirectedBoostGraph::getNumNodes() {
  return num_vertices(*dag_);
}


std::vector<VertexIdType> DirectedBoostGraph::getNeighborList(VertexIdType vertex_id) {
  std::vector<VertexIdType> l;

  typedef typename boost::property_map<d_adj_list, boost::vertex_index_t>::type IndexMap;
  IndexMap indexMap = get(boost::vertex_index, *dag_);

  typedef typename boost::graph_traits<d_adj_list>::adjacency_iterator adjacency_iterator;

  std::pair<adjacency_iterator, adjacency_iterator> neighbors =
    boost::adjacent_vertices(vertex(vertex_id, *dag_), *dag_);

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
           get(edge_weight, *dag_);
  std::vector<VertexIdType> p(num_vertices(*dag_));
  std::vector<std::size_t> d(num_vertices(*dag_));
  d_vertex_type s = vertex(startIndex, *dag_);

  dijkstra_shortest_paths(
      *dag_, s,
      predecessor_map(boost::make_iterator_property_map(
                             p.begin(), get(boost::vertex_index, *dag_)))
          .distance_map(boost::make_iterator_property_map(
                               d.begin(), get(boost::vertex_index, *dag_))));

  for (const auto & di: d) distances.push_back(static_cast<double>(di));
  for (const auto & pi: p) paths.push_back(pi);
  return;
}

} // namespace runtime
} // namespace exatn
