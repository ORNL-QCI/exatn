/** ExaTN::Numerics: Tensor contraction sequence optimizer: Base
REVISION: 2020/08/06

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "contraction_seq_optimizer.hpp"
#include "tensor_network.hpp"
#include "metis_graph.hpp"

namespace exatn{

namespace numerics{

//Cache of already determined tensor network contraction sequences:
std::unordered_map<std::string,ContractionSeqOptimizer::CachedContrSeq> ContractionSeqOptimizer::cached_contr_seqs_;


void packContractionSequenceIntoVector(const std::list<ContrTriple> & contr_sequence,
                                       std::vector<unsigned int> & contr_sequence_content)
{
 const auto num_contractions = contr_sequence.size();
 contr_sequence_content.resize(num_contractions*3);
 std::size_t i = 0;
 for(const auto & contr: contr_sequence){
  contr_sequence_content[i++] = contr.result_id;
  contr_sequence_content[i++] = contr.left_id;
  contr_sequence_content[i++] = contr.right_id;
 }
 return;
}


void unpackContractionSequenceFromVector(std::list<ContrTriple> & contr_sequence,
                                         const std::vector<unsigned int> & contr_sequence_content)
{
 assert(contr_sequence_content.size() % 3 == 0);
 const auto num_contractions = contr_sequence_content.size() / 3;
 contr_sequence.resize(num_contractions);
 std::size_t i = 0;
 for(auto iter = contr_sequence.begin(); iter != contr_sequence.end(); ++iter){
  iter->result_id = contr_sequence_content[i++];
  iter->left_id = contr_sequence_content[i++];
  iter->right_id = contr_sequence_content[i++];
 }
 return;
}


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
