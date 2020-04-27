/** ExaTN::Numerics: Graph k-way partitioning via METIS
REVISION: 2020/04/27

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
**/

#ifndef EXATN_NUMERICS_METIS_GRAPH_HPP_
#define EXATN_NUMERICS_METIS_GRAPH_HPP_

#include "metis.h"

#include "tensor_basic.hpp"

#include <vector>

namespace exatn{

namespace numerics{

class TensorNetwork;


class MetisGraph{

public:

 /** Constructs an empty graph. **/
 MetisGraph();

 /** Constructs a graph from an existing tensor network.
     Original tensor Ids will be stored as well in order
     to retrieve graph partitions in terms of the original
     tensor Ids later. **/
 MetisGraph(const TensorNetwork & network);

 MetisGraph(const MetisGraph &) = default;
 MetisGraph & operator=(const MetisGraph &) = default;
 MetisGraph(MetisGraph &&) noexcept = default;
 MetisGraph & operator=(MetisGraph &&) noexcept = default;
 ~MetisGraph() = default;

 /** Clears back to an empty state. **/
 void clear();

 /** Appends a new vertex with its edges. **/
 void appendVertex(std::size_t num_edges,      //in: number of edges (number of adjacent vertices)
                   std::size_t * adj_vertices, //in: adjacent vertices (numbering starts from 0)
                   std::size_t * edge_weights, //in: edge weights (>0)
                   std::size_t vertex_weight); //in: vertex weight (>=0)

 /** Returns the cost of contraction of two vertices. **/
 double getContractionCost(std::size_t vertex1, //in: first vertex id [0..N-1]
                           std::size_t vertex2, //in: second vertex id [0..N-1]
                           double * intermediate_volume = nullptr, //out: volume of the produced intermediate
                           double * diff_volume = nullptr) const;  //out: differential volume (intermediate volume - input volumes)

 /** Contracts (merges) two vertices. The resulting aggregated
     vertex will replace vertex1 (vertex1 < vertex 2). **/
 bool mergeVertices(std::size_t vertex1,  //in: first vertex id [0..N-1]
                    std::size_t vertex2); //in: second vertex id [0..N-1]

 /** Partitions the graph into a desired number of partitions while
     minimizing the weighted edge cut. **/
 bool partitionGraph(std::size_t num_parts, //in: number of partitions
                     double imbalance);     //in: partition size imbalance tolerance (>= 1.0)

 /** Retrieves the graph partitions and the achieved edge cut value.
     The renumbering vector returns the new-to-old vertex id mapping, if any. **/
 const std::vector<idx_t> & getPartitions(std::size_t * edge_cut = nullptr,
                                          std::size_t * num_cross_edges = nullptr,
                                          const std::vector<idx_t> ** renumbering = nullptr) const;

protected:

 void initMetisGraph();

private:

 idx_t options_[METIS_NOPTIONS];   //METIS options

 idx_t num_vertices_;              //number of vertices in the graph
 std::vector<idx_t> renumber_;     //new vertex id --> old vertex id (if renumbering was done)
 std::vector<idx_t> xadj_,adjncy_; //graph structure stored in the CSR format
 std::vector<idx_t> vwgt_;         //vertex weights
 std::vector<idx_t> adjwgt_;       //edge weights

 idx_t num_parts_;               //number of parts in graph partitioning
 std::vector<real_t> tpwgts_;    //desired weight of each partition (adds up to 1.0)
 std::vector<real_t> ubvec_;     //imbalance tolerance for each partition (>= 1.0)
 std::vector<idx_t> partitions_; //computed partitions
 idx_t edge_cut_;                //achieved edge cut
 idx_t num_cross_edges_;         //number of cross edges in the edge cut
};


} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_METIS_GRAPH_HPP_
