/** ExaTN::Numerics: Graph k-way partitioning via METIS
REVISION: 2020/04/25

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "metis_graph.hpp"

#include <iostream>

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

void MetisGraph::clear()
{
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

void MetisGraph::appendEdges(std::size_t num_edges,      //in: number of edges (number of adjacent vertices)
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

bool MetisGraph::partitionGraph(std::size_t num_parts, //in: number of parts (>0)
                                double imbalance)      //in: imbalance tolerance (>= 1.0)
{
 assert(num_parts > 0);
 assert(imbalance >= 1.0);
 assert(num_vertices_ > 0);
 num_parts_ = num_parts;
 idx_t ncon = 1;
 auto errc = METIS_PartGraphKway(&num_vertices_,&ncon,xadj_.data(),adjncy_.data(),
                                 vwgt_.data(),NULL,adjwgt_.data(),&num_parts_,
                                 NULL,&imbalance,options_,&edge_cut_,partitions_.data());
 if(errc != METIS_OK) std::cout << "#ERROR(exatn::numerics::MetisGraph): METIS error "
                                << errc << std::endl;
 return (errc == METIS_OK);
}

const std::vector<idx_t> & MetisGraph::getPartitions(std::size_t * edge_cut) const
{
 assert(edge_cut != nullptr);
 *edge_cut = edge_cut_;
 return partitions_;
}

} //namespace numerics

} //namespace exatn
