/** ExaTN::Numerics: Graph k-way partitioning via METIS
REVISION: 2020/04/26

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "metis_graph.hpp"

#include "tensor_network.hpp"

#include <iostream>
#include <unordered_map>
#include <algorithm>

#include <cassert>

namespace exatn{

namespace numerics{

void MetisGraph::initMetisGraph()
{
 METIS_SetDefaultOptions(options_);
 options_[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
 options_[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
 options_[METIS_OPTION_NUMBERING] = 0;
 options_[METIS_OPTION_MINCONN] = 1;
 options_[METIS_OPTION_CONTIG] = 0;
 options_[METIS_OPTION_CCORDER] = 1;

 num_vertices_ = 0;
 num_parts_ = 0;
 edge_cut_ = 0;
 xadj_.emplace_back(0);
 return;
}


MetisGraph::MetisGraph():
 num_vertices_(0), num_parts_(0), edge_cut_(0)
{
 initMetisGraph();
}


MetisGraph::MetisGraph(const TensorNetwork & network):
 MetisGraph()
{
 //Map tensor Ids to a consecutive integer range [0..N-1], N is the number of input tensors:
 std::unordered_map<unsigned int, unsigned int> tensor_id_map; //tensor_id --> vertex_id [0..N-1]
 unsigned int vertex_id = 0;
 for(auto iter = network.cbegin(); iter != network.cend(); ++iter){
  if(iter->first != 0){ //ignore the output tensor
   renumber_.emplace_back(iter->first);
   auto res = tensor_id_map.emplace(std::make_pair(iter->first,vertex_id++));
   assert(res.second);
  }
 }
 //Generate the adjacency list:
 auto cmp = [](std::pair<unsigned int, DimExtent> & a,
               std::pair<unsigned int, DimExtent> & b){return (a.first < b.first);};
 for(auto iter = network.cbegin(); iter != network.cend(); ++iter){
  if(iter->first != 0){ //ignore the output tensor
   const auto tensor_rank = iter->second.getRank();
   const auto & tensor_dims = iter->second.getDimExtents();
   const auto & tensor_legs = iter->second.getTensorLegs();
   std::vector<std::pair<unsigned int, DimExtent>> edges(tensor_rank);
   for(unsigned int i = 0; i < tensor_rank; ++i){
    edges[i] = std::pair<unsigned int, DimExtent>{tensor_legs[i].getTensorId(),
                                                  tensor_dims[i]};
   }
   //Remap tensor id to the vertex id:
   std::sort(edges.begin(),edges.end(),cmp);
   std::size_t adj_vertices[tensor_rank], edge_weights[tensor_rank];
   std::size_t vertex_weight = 1;
   unsigned int first_vertex_pos = tensor_rank;
   for(unsigned int i = 0; i < tensor_rank; ++i){
    if(edges[i].first == 0){ //connections to the output tensor are not counted as edges
     vertex_weight *= edges[i].second; //they are absorbed into the vertex weight
    }else{
     if(first_vertex_pos == tensor_rank) first_vertex_pos = i;
     edges[i].first = tensor_id_map[edges[i].first];
    }
   }
   //Compute edge weights and adjacency list:
   std::size_t num_edges = 0;
   std::size_t edge_weight = 1;
   int current_vertex = -1;
   for(int i = first_vertex_pos; i < tensor_rank; ++i){
    if(edges[i].first != current_vertex){
     if(current_vertex >= 0){
      adj_vertices[num_edges] = current_vertex;
      edge_weights[num_edges++] = edge_weight;
     }
     current_vertex = edges[i].first;
     edge_weight = edges[i].second;
    }else{
     edge_weight *= edges[i].second;
    }
   }
   if(current_vertex >= 0){
    adj_vertices[num_edges] = current_vertex;
    edge_weights[num_edges++] = edge_weight;
   }
   appendVertex(num_edges,adj_vertices,edge_weights,vertex_weight);
  }
 }
}


void MetisGraph::clear()
{
 renumber_.clear();
 xadj_.clear();
 adjncy_.clear();
 vwgt_.clear();
 adjwgt_.clear();
 tpwgts_.clear();
 ubvec_.clear();
 partitions_.clear();
 initMetisGraph();
 return;
}


void MetisGraph::appendVertex(std::size_t num_edges,      //in: number of edges (number of adjacent vertices)
                              std::size_t * adj_vertices, //in: adjacent vertices (numbering starts from 0)
                              std::size_t * edge_weights, //in: edge weights (>0)
                              std::size_t vertex_weight)  //in: vertex weight (>=0)
{
 for(std::size_t i = 0; i < num_edges; ++i) adjncy_.emplace_back(adj_vertices[i]);
 for(std::size_t i = 0; i < num_edges; ++i) adjwgt_.emplace_back(edge_weights[i]);
 xadj_.emplace_back(xadj_[num_vertices_] + num_edges);
 vwgt_.emplace_back(vertex_weight);
 ++num_vertices_;
 return;
}


double MetisGraph::getContractionCost(std::size_t vertex1, //in: first vertex id [0..N-1]
                                      std::size_t vertex2, //in: second vertex id [0..N-1]
                                      double * intermediate_volume, //out: volume of the produced intermediate
                                      double * diff_volume) const   //out: differential volume (intermediate volume - input volumes)
{
 double flops = 0.0;
 if(vertex1 != vertex2 && vertex1 < num_vertices_ && vertex2 < num_vertices_){
  if(vertex1 > vertex2) std::swap(vertex1,vertex2);
  double left_vol = vwgt_[vertex1];  //open volume of vertex1
  double right_vol = vwgt_[vertex2]; //open volume of vertex2
  double contr_vol = 1.0;
  for(auto i = xadj_[vertex1]; i < xadj_[vertex1+1]; ++i){ //edges of vertex1
   double vol = static_cast<double>(adjwgt_[i]);
   if(adjncy_[i] == vertex2) contr_vol *= vol;
   left_vol *= vol;
  }
  for(auto i = xadj_[vertex2]; i < xadj_[vertex2+1]; ++i){ //edges of vertex2
   double vol = static_cast<double>(adjwgt_[i]);
   right_vol *= vol;
  }
  double inter_vol = left_vol * right_vol / (contr_vol * contr_vol);
  if(intermediate_volume != nullptr) *intermediate_volume = inter_vol;
  if(diff_volume != nullptr) *diff_volume = inter_vol - (left_vol + right_vol);
 }
 return flops;
}


bool MetisGraph::mergeVertices(std::size_t vertex1, //in: first vertex id [0..N-1]
                               std::size_t vertex2) //in: second vertex id [0..N-1]
{
 assert(xadj_.size() == (num_vertices_ + 1) && vwgt_.size() == num_vertices_);
 if(vertex1 == vertex2 || vertex1 >= num_vertices_ || vertex2 >= num_vertices_) return false;
 if(vertex1 > vertex2) std::swap(vertex1,vertex2);
 //`Finish
 vwgt_[vertex1] *= vwgt_[vertex2]; //combine open volumes into vertex1
 vwgt_.erase(vwgt_.begin()+vertex2); //delete vertex2
 --num_vertices_;
 return true;
}


bool MetisGraph::partitionGraph(std::size_t num_parts, //in: number of parts (>0)
                                double imbalance)      //in: imbalance tolerance (>= 1.0)
{
 assert(num_parts > 0);
 assert(imbalance >= 1.0);
 assert(num_vertices_ > 0);
 num_parts_ = num_parts;
 partitions_.resize(num_vertices_);
 idx_t ncon = 1;
 auto errc = METIS_PartGraphKway(&num_vertices_,&ncon,xadj_.data(),adjncy_.data(),
                                 vwgt_.data(),NULL,adjwgt_.data(),&num_parts_,
                                 NULL,&imbalance,options_,&edge_cut_,partitions_.data());
 if(errc != METIS_OK){
  std::cout << "#ERROR(exatn::numerics::MetisGraph): METIS error " << errc << std::endl;
 }
 return (errc == METIS_OK);
}


const std::vector<idx_t> & MetisGraph::getPartitions(std::size_t * edge_cut,
                                                     const std::vector<idx_t> ** renumbering) const
{
 if(edge_cut != nullptr) *edge_cut = edge_cut_;
 if(renumbering != nullptr){
  if(renumber_.empty()){
   *renumbering = nullptr;
  }else{
   *renumbering = &renumber_;
  }
 }
 return partitions_;
}

} //namespace numerics

} //namespace exatn
