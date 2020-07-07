/** ExaTN::Numerics: Tensor contraction sequence optimizer: Base
REVISION: 2020/07/06

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "contraction_seq_optimizer.hpp"
#include "tensor_network.hpp"
#include "metis_graph.hpp"

namespace exatn{

namespace numerics{

std::unordered_map<std::string,std::pair<MetisGraph,std::list<ContrTriple>>> ContractionSeqOptimizer::cached_contr_seqs_;


bool ContractionSeqOptimizer::cacheContractionSequence(const TensorNetwork & network)
{
 auto res = cached_contr_seqs_.emplace(network.getName(),
            std::make_pair(MetisGraph(network),network.exportContractionSequence()));
 return res.second;
}


bool ContractionSeqOptimizer::eraseContractionSequence(const TensorNetwork & network)
{
 auto num_deleted = cached_contr_seqs_.erase(network.getName());
 return (num_deleted == 1);
}


const std::list<ContrTriple> * ContractionSeqOptimizer::findContractionSequence(const TensorNetwork & network)
{
 auto iter = cached_contr_seqs_.find(network.getName());
 if(iter != cached_contr_seqs_.end()){
  MetisGraph network_graph(network);
  if(network_graph == iter->second.first) return &(iter->second.second);
 };
 return nullptr;
}

} //namespace numerics

} //namespace exatn
