/** ExaTN::Numerics: Tensor contraction sequence optimizer: Base
REVISION: 2021/04/01

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "contraction_seq_optimizer.hpp"
#include "tensor_network.hpp"
#include "metis_graph.hpp"

#include <iostream>
#include <fstream>

namespace exatn{

namespace numerics{

//Cache of already determined tensor network contraction sequences:
std::unordered_map<std::string,ContractionSeqOptimizer::CachedContrSeq> ContractionSeqOptimizer::cached_contr_seqs_;
bool ContractionSeqOptimizer::cache_to_disk_{false};


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
  if(res.second && cache_to_disk_){
   const auto & kv = *(res.first);
   std::ofstream cseq_file(kv.first + ".cseq.exatn",std::ios::out|std::ios::trunc);
   cseq_file << kv.second.fma_flops << " " << kv.second.contr_seq.size() << std::endl;
   for(const auto & triple: kv.second.contr_seq){
    cseq_file << triple.result_id << " " << triple.left_id << " " << triple.right_id << std::endl;
   }
   cseq_file.close();
  }
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
 }else{
  if(cache_to_disk_){
   std::ifstream cseq_file(network.getName() + ".cseq.exatn",std::ios::in);
   if(cseq_file.is_open()){
    double flops = 0.0;
    std::size_t num_contractions = 0;
    cseq_file >> flops >> num_contractions;
    //std::cout << "#DEBUG: Reading cseq.exatn file: " << flops << " " << num_contractions << std::endl; //debug
    auto res = cached_contr_seqs_.emplace(network.getName(),
     std::move(CachedContrSeq{std::make_shared<MetisGraph>(network),std::list<ContrTriple>(num_contractions),flops}));
    if(res.second){
     auto & cseq = res.first->second.contr_seq;
     for(auto contr = cseq.begin(); contr != cseq.end(); ++contr){
      cseq_file >> contr->result_id >> contr->left_id >> contr->right_id;
     }
     cseq_file.close();
     return std::make_pair(&(res.first->second.contr_seq),res.first->second.fma_flops);
    }else{
     cseq_file.close();
    }
   }
  }
 }
 return std::pair<const std::list<ContrTriple> *, double> {nullptr,0.0};
}


void ContractionSeqOptimizer::activatePersistentCaching(bool persist)
{
 cache_to_disk_ = persist;
 return;
}

} //namespace numerics

} //namespace exatn
