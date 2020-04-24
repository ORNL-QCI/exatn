/** ExaTN::Numerics: Graph k-way partitioning via METIS
REVISION: 2020/04/24

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

class MetisGraph{

public:

 MetisGraph();

 /** Clears back to an empty state. **/
 void clear();

 /** Appends edges of a new vertex and increments the number of vertices. **/
 void appendEdges();

 /** Partitions the graph into a desired number of parts. **/
 bool partitionGraph(std::size_t num_parts, //in: number of parts
                     double imbalance);     //in: imbalance tolerance (>= 1.0)

 /** Retrieves the partitions. **/
 const std::vector<idx_t> & getPartitions() const;

private:

 idx_t options_[METIS_NOPTIONS];   //METIS options

 idx_t num_vertices_;              //number of vertices in the graph
 std::vector<idx_t> xadj_,adjncy_; //graph structure stored in the CSR format
 std::vector<idx_t> vwgt_;         //vertex weights
 std::vector<idx_t> adjwgt_;       //edge weights

 idx_t num_parts_;               //number of parts in graph partitioning
 std::vector<real_t> tpwgts_;    //desired weight of each partition (adds up to 1.0)
 std::vector<real_t> ubvec_;     //imbalance tolerance for each partition (>= 1.0)
 std::vector<idx_t> partitions_; //partitioning
};


} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_METIS_GRAPH_HPP_
