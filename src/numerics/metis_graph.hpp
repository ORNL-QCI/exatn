/** ExaTN::Numerics: Graph k-way partitioning via METIS
REVISION: 2020/04/29

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) METIS graph is a weighted undirectional simple graph where both
     vertices and edges have positive integer weights.
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
     tensor Ids later via the getPartitions() method.
     The output tensor of the tensor network does not form
     a vertex and all connections to the output tensor
     originating from a given input tensor are aggregated and
     incorporated as the weight of the corresponding vertex.
     Multiple connections between two input tensors are
     combined into an aggregated single edge with the weight
     equal to the total volume of these aggregated connections.
     Note that multiple connections between two tensors are
     aggregated into the vertex/edge weight logarithmically. **/
 MetisGraph(const TensorNetwork & network); //in: tensor network

 /** Constructs a graph from a given partition of a larger graph.
     The open edges are aggregated into the weights of incident vertices. **/
 MetisGraph(const MetisGraph & parent, //in: partitioned parental graph
            std::size_t partition);    //in: partition of the parental graph

 MetisGraph(const MetisGraph &) = default;
 MetisGraph & operator=(const MetisGraph &) = default;
 MetisGraph(MetisGraph &&) noexcept = default;
 MetisGraph & operator=(MetisGraph &&) noexcept = default;
 ~MetisGraph() = default;

 /** Clears the current partitioning. **/
 void clearPartitions();

 /** Clears the graph back to an empty state. **/
 void clear();

 /** Returns the number of vertices in the graph. **/
 std::size_t getNumVertices() const;

 /** Returns the number of partitions the most recent partitioning
     resulted in, or zero otherwise. **/
 std::size_t getNumPartitions() const;

 /** Appends a new vertex with its edges. **/
 void appendVertex(std::size_t num_edges,      //in: number of edges (number of adjacent vertices)
                   std::size_t * adj_vertices, //in: adjacent vertices (numbering starts from 0)
                   std::size_t * edge_weights, //in: edge weights (>0)
                   std::size_t vertex_weight); //in: vertex weight (>=0)

 /** Returns an approximate cost of contraction of two vertices (FMA flops). **/
 double getContractionCost(std::size_t vertex1, //in: first vertex id [0..N-1]
                           std::size_t vertex2, //in: second vertex id [0..N-1]
                           double * intermediate_volume = nullptr, //out: approximate volume of the produced intermediate
                           double * diff_volume = nullptr) const;  //out: approximate differential volume (intermediate volume - input volumes)

 /** Contracts (merges) two vertices. The resulting aggregated
     vertex will replace vertex1 (vertex1 < vertex 2). **/
 bool mergeVertices(std::size_t vertex1,  //in: first vertex id [0..N-1]
                    std::size_t vertex2); //in: second vertex id [0..N-1]

 /** Partitions the graph into a desired number of partitions while minimizing
     the weighted edge cut under tolerated partition weight imbalance. **/
 bool partitionGraph(std::size_t num_parts, //in: number of partitions (>0)
                     double imbalance);     //in: partition weight imbalance tolerance (>= 1.0)

 /** Retrieves the graph partitions and the achieved edge cut value.
     The renumbering vector returns the new-to-old vertex id mapping for
     cases when the graph was created from a tensor network di-multi-graph. **/
 const std::vector<idx_t> & getPartitions(std::size_t * edge_cut = nullptr, //out: achieved edge cut value
                                          std::size_t * num_cross_edges = nullptr, //out: total number of cross edges
                                          const std::vector<idx_t> ** part_weights = nullptr, //out: partition weights
                                          const std::vector<idx_t> ** renumbering = nullptr) const; //out: vertex id renumbering

 /** Returns the original id for a given graph vertex. **/
 std::size_t getOriginalVertexId(std::size_t vertex_id) const; //in: current vertex id [0..N-1]

protected:

 void initMetisGraph();

private:
 //METIS options:
 idx_t options_[METIS_NOPTIONS];   //METIS options
 //Graph structure:
 idx_t num_vertices_;              //number of vertices in the graph
 std::vector<idx_t> renumber_;     //new vertex id --> old vertex id (if renumbering was done)
 std::vector<idx_t> xadj_,adjncy_; //graph structure stored in the CSR format
 std::vector<idx_t> vwgt_;         //vertex weights
 std::vector<idx_t> adjwgt_;       //edge weights
 //Partitioning:
 idx_t num_parts_;                 //number of parts in graph partitioning
 std::vector<real_t> tpwgts_;      //desired weight of each partition (adds up to 1.0)
 std::vector<real_t> ubvec_;       //weight imbalance tolerance for each partition (>= 1.0)
 std::vector<idx_t> partitions_;   //computed partitions
 std::vector<idx_t> part_weights_; //actual partition weights
 idx_t edge_cut_;                  //achieved edge cut
 idx_t num_cross_edges_;           //number of cross edges in the edge cut
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_METIS_GRAPH_HPP_
