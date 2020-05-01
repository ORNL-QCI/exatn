/** ExaTN::Numerics: Tensor contraction sequence optimizer: Metis heuristics
REVISION: 2020/04/30

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "contraction_seq_optimizer_metis.hpp"
#include "tensor_network.hpp"

#include "metis_graph.hpp"

#include <cassert>

#include <deque>
#include <tuple>
#include <chrono>

namespace exatn{

namespace numerics{

ContractionSeqOptimizerMetis::ContractionSeqOptimizerMetis():
 num_walkers_(NUM_WALKERS), acceptance_tolerance_(ACCEPTANCE_TOLERANCE),
 partition_factor_(PARTITION_FACTOR), partition_max_size_(PARTITION_MAX_SIZE),
 partition_imbalance_(PARTITION_IMBALANCE)
{
}


void ContractionSeqOptimizerMetis::resetNumWalkers(unsigned int num_walkers)
{
 num_walkers_ = num_walkers;
 return;
}


void ContractionSeqOptimizerMetis::resetAcceptanceTolerance(double acceptance_tolerance)
{
 acceptance_tolerance_ = acceptance_tolerance;
 return;
}


double ContractionSeqOptimizerMetis::determineContractionSequence(const TensorNetwork & network,
                                                                  std::list<ContrTriple> & contr_seq,
                                                                  std::function<unsigned int ()> intermediate_num_generator)
{
 const bool debugging = false;

 double flops = 0.0;
 contr_seq.clear();
 auto num_contractions = (network.getNumTensors() - 1); //number of contractions is one less than the number of r.h.s. tensors
 if(num_contractions == 0) return flops;

 //Search for the optimal tensor contraction sequence:
 if(debugging) std::cout << "#DEBUG(ContractionSeqOptimizerMetis): Searching for a pseudo-optimal tensor contraction sequence:\n"; //debug
 const auto partition_imbalance_original = partition_imbalance_;
 bool not_done = true;
 while(not_done){
  //Determine a tensor contraction sequence:
  std::list<ContrTriple> cseq;
  determineContrSequence(network,cseq,intermediate_num_generator);
  //Compute the total FMA flop count:
  TensorNetwork net(network);
  double flps = 0.0;
  for(const auto & contr_triple: cseq){
   flps += net.getContractionCost(contr_triple.left_id,contr_triple.right_id);
   if(contr_triple.result_id != 0){ //intermediate tensor contraction
    bool success = net.mergeTensors(contr_triple.left_id,contr_triple.right_id,contr_triple.result_id);
    assert(success);
   }else{ //last tensor contraction (into the output tensor)
    assert(net.getNumTensors() == 2);
   }
  }
  //Compare with previous best:
  if(flops > 0.0){
   if(flops > flps){
    contr_seq = cseq;
    flops = flps;
    if(debugging) std::cout << " A better tensor contraction sequence found with Flop count = " << flops
                            << " under imbalance = " << partition_imbalance_ << std::endl; //debug
   }
  }else{
   contr_seq = cseq;
   flops = flps;
   if(debugging) std::cout << " A better tensor contraction sequence found with Flop count = " << flops
                           << " under imbalance = " << partition_imbalance_ << std::endl; //debug
  }
  //Next iteration:
  cseq.clear();
  partition_imbalance_ += 0.01;
  not_done = (partition_imbalance_ < 2.0);
 }
 partition_imbalance_ = partition_imbalance_original;
 if(debugging) std::cout << "#DEBUG(ContractionSeqOptimizerMetis): The pseudo-optimal Flop count found = " << flops << std::endl; //debug
 return flops;
}


void ContractionSeqOptimizerMetis::determineContrSequence(const TensorNetwork & network,
                                                          std::list<ContrTriple> & contr_seq,
                                                          std::function<unsigned int ()> intermediate_num_generator)
{
 const bool debugging = false;

 contr_seq.clear();
 auto num_contractions = (network.getNumTensors() - 1); //number of contractions is one less than the number of r.h.s. tensors
 if(num_contractions == 0) return;

 if(debugging) std::cout << "#DEBUG(ContractionSeqOptimizerMetis): Determining a pseudo-optimal tensor contraction sequence ... \n"; //debug
 auto time_beg = std::chrono::high_resolution_clock::now();

 //Recursive Kway partitioning:
 std::deque<std::pair<MetisGraph,   //graph of a tensor sub-network
                      unsigned int> //tensor id for the intermediate output tensor of the sub-network
           > graphs; //graphs of tensor sub-networks
 graphs.emplace_back(std::make_pair(MetisGraph(network),0)); //original full tensor network graph
 bool not_done = true;
 while(not_done){
  not_done = false;
  auto num_graphs_in = graphs.size();
  for(std::size_t i = 0; i < num_graphs_in; ++i){
   auto & graph = graphs[i].first; //parent graph
   if(graph.getNumVertices() > partition_max_size_){
    not_done = true;
    bool success = graph.partitionGraph(partition_factor_,partition_imbalance_); assert(success);
    const auto num_partitions = graph.getNumPartitions(); assert(num_partitions == 2);
    for(std::size_t j = 0; j < num_partitions; ++j){
     graphs.emplace_back(std::make_pair(MetisGraph(graph,j),
                                        intermediate_num_generator())); //child graphs
    }
    const auto last_tensor_pos = graphs.size() - 1;
    contr_seq.emplace_front(ContrTriple{graphs[i].second,
                                        graphs[last_tensor_pos-1].second,
                                        graphs[last_tensor_pos].second});
   }else{
    graphs.emplace_back(graphs[i]); //parent graph is already small enough, no further division
   }
  }
  while(num_graphs_in-- > 0) graphs.pop_front(); //delete parent graphs
 }
 //Append the very first tensor contractions:
 for(const auto & graph_entry: graphs){
  const auto & graph = graph_entry.first;
  const auto num_vertices = graph.getNumVertices();
  if(num_vertices == 3){
   unsigned int result_id = intermediate_num_generator();
   unsigned int right_id = graph.getOriginalVertexId(2);
   contr_seq.emplace_front(ContrTriple{graph_entry.second,result_id,right_id});
   unsigned int left_id = graph.getOriginalVertexId(0);
   right_id = graph.getOriginalVertexId(1);
   contr_seq.emplace_front(ContrTriple{result_id,left_id,right_id});
  }else if(num_vertices == 2){
   unsigned int left_id = graph.getOriginalVertexId(0);
   unsigned int right_id = graph.getOriginalVertexId(1);
   contr_seq.emplace_front(ContrTriple{graph_entry.second,left_id,right_id});
  }else{
   unsigned int intermediate_id = graph_entry.second;
   unsigned int tensor_id = graph.getOriginalVertexId(0);
   for(auto & contr_triple: contr_seq){
    if(contr_triple.left_id == intermediate_id) contr_triple.left_id = tensor_id;
    if(contr_triple.right_id == intermediate_id) contr_triple.right_id = tensor_id;
   }
  }
 }
 assert(contr_seq.size() == num_contractions);

 auto time_end = std::chrono::high_resolution_clock::now();
 auto time_total = std::chrono::duration_cast<std::chrono::duration<double>>(time_end - time_beg);
 if(debugging){
  std::cout << "#DEBUG(ContractionSeqOptimizerMetis): Done (" << time_total.count() << " sec):"; //debug
  for(const auto & contr_pair: contr_seq) std::cout << " {" << contr_pair.left_id << ","
                                                            << contr_pair.right_id << "->"
                                                            << contr_pair.result_id << "}"; //debug
  std::cout << std::endl; //debug
 }
 return;
}


std::unique_ptr<ContractionSeqOptimizer> ContractionSeqOptimizerMetis::createNew()
{
 return std::unique_ptr<ContractionSeqOptimizer>(new ContractionSeqOptimizerMetis());
}

} //namespace numerics

} //namespace exatn
