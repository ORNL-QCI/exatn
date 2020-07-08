/** ExaTN::Numerics: Tensor contraction sequence optimizer: Base
REVISION: 2020/07/08

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "contraction_seq_optimizer.hpp"
#include "tensor_network.hpp"
#include "metis_graph.hpp"

namespace exatn{

namespace numerics{

std::unordered_map<std::string,ContractionSeqOptimizer::CachedContrSeq> ContractionSeqOptimizer::cached_contr_seqs_;


bool ContractionSeqOptimizer::cacheContractionSequence(const TensorNetwork & network)
{
 if(!(network.exportContractionSequence().empty())){
  auto res = cached_contr_seqs_.emplace(network.getName(),
   std::move(CachedContrSeq{std::make_shared<MetisGraph>(network),network.exportContractionSequence(),network.getFMAFlops()}));
  return res.second;
 }
 return false;
}


bool ContractionSeqOptimizer::eraseContractionSequence(const TensorNetwork & network)
{
 auto num_deleted = cached_contr_seqs_.erase(network.getName());
 return (num_deleted == 1);
}


std::pair<const std::list<ContrTriple> *, double> ContractionSeqOptimizer::findContractionSequence(const TensorNetwork & network)
{
 auto iter = cached_contr_seqs_.find(network.getName());
 if(iter != cached_contr_seqs_.end()){
  MetisGraph network_graph(network);
  if(network_graph == *(iter->second.graph)) return std::make_pair(&(iter->second.contr_seq),iter->second.fma_flops);
 };
 return std::pair<const std::list<ContrTriple> *, double> {nullptr,0.0};
}

} //namespace numerics

} //namespace exatn
