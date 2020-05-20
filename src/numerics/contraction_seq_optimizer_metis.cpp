/** ExaTN::Numerics: Tensor contraction sequence optimizer: Metis heuristics
REVISION: 2020/05/20

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "contraction_seq_optimizer_metis.hpp"
#include "tensor_network.hpp"

#include "metis_graph.hpp"

#include <algorithm>
#include <random>
#include <deque>
#include <tuple>
#include <chrono>

#include <cmath>
#include <cassert>

namespace exatn{

namespace numerics{

constexpr const double ContractionSeqOptimizerMetis::PARTITION_IMBALANCE;


ContractionSeqOptimizerMetis::ContractionSeqOptimizerMetis():
 num_walkers_(NUM_WALKERS), acceptance_tolerance_(ACCEPTANCE_TOLERANCE),
 partition_factor_(PARTITION_FACTOR), partition_granularity_(PARTITION_GRANULARITY),
 partition_max_size_(PARTITION_MAX_SIZE),
 partition_imbalance_(PARTITION_IMBALANCE_DEPTH,PARTITION_IMBALANCE)
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
 const bool deterministic = false;

 double flops = 0.0;
 contr_seq.clear();
 std::size_t num_tensors = network.getNumTensors();
 auto num_contractions = (num_tensors - 1); //number of contractions is one less than the number of r.h.s. tensors
 if(num_contractions == 0) return flops;

 //Search for the optimal tensor contraction sequence:
 if(debugging) std::cout << "#DEBUG(ContractionSeqOptimizerMetis): Searching for a pseudo-optimal tensor contraction sequence:\n"; //debug
 std::random_device seeder;
 std::default_random_engine generator(seeder());
 std::uniform_real_distribution<double> distribution(1.001,1.999);
 auto rnd = std::bind(distribution,generator);

 double max_flop = 0.0;
 partition_granularity_ = std::max(partition_factor_,std::min(partition_granularity_,num_tensors/(2*partition_max_size_)));
 while(partition_granularity_ >= partition_factor_){
  auto num_walkers = num_walkers_;
  while(num_walkers-- > 0){
   //Determine a tensor contraction sequence:
   std::list<ContrTriple> cseq;
   determineContrSequence(network,cseq,intermediate_num_generator);
   //Compute the total FMA flop count:
   TensorNetwork net(network);
   std::vector<double> contr_flops(cseq.size(),0.0);
   double flps = 0.0; std::size_t i = 0;
   for(const auto & contr_triple: cseq){
    contr_flops[i] = net.getContractionCost(contr_triple.left_id,contr_triple.right_id);
    flps += contr_flops[i++];
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
     max_flop = *(std::max_element(contr_flops.cbegin(),contr_flops.cend()));
     contr_seq = cseq;
     flops = flps;
     if(debugging){
      std::cout << " Iteration " << (num_walkers + 1)
                << ": A faster tensor contraction sequence found with Flop count = " << flops
                << " with top granularity " << partition_granularity_ << " under imbalances:";
      for(const auto & imbalance: partition_imbalance_) std::cout << " " << imbalance;
      std::cout << ":\n";
      for(const auto & contr_cost: contr_flops) std::cout << " " << contr_cost;
      std::cout << std::endl;
     }
     num_walkers = num_walkers_;
    }
   }else{
    max_flop = *(std::max_element(contr_flops.cbegin(),contr_flops.cend()));
    contr_seq = cseq;
    flops = flps;
    if(debugging){
     std::cout << " Iteration " << (num_walkers + 1)
               << ": A faster tensor contraction sequence found with Flop count = " << flops
               << " with top granularity " << partition_granularity_ << " under imbalances:";
     for(const auto & imbalance: partition_imbalance_) std::cout << " " << imbalance;
     std::cout << ":\n";
     for(const auto & contr_cost: contr_flops) std::cout << " " << contr_cost;
     std::cout << std::endl;
    }
   }
   //Update partition imbalances:
   if(deterministic){ //deterministic update
    auto adjust_func = [](double z, double x){return z/(z + (1.0 - z) * std::exp(-0.33 * x));};
    auto contr = cseq.size(); //last tensor contraction
    for(auto & imbalance: partition_imbalance_){
     const double diff = std::log10(contr_flops[--contr]) - std::log10(max_flop);
     imbalance = std::pow(2.0,adjust_func(std::log2(imbalance),diff));
     if(contr == 0) break;
    }
   }else{ //random update
    for(auto & imbalance: partition_imbalance_) imbalance = rnd();
   }
   /*
   if(debugging){
    std::cout << " Updated imbalances: ";
    for(const auto & imbalance: partition_imbalance_) std::cout << " " << imbalance;
    std::cout << std::endl;
    std::cout << " Reverse contraction costs: ";
    contr = cseq.size();
    for(const auto & imbalance: partition_imbalance_){
     std::cout << " " << contr_flops[--contr];
     if(contr == 0) break;
    }
    std::cout << std::endl;
   }
   */
  }
  --partition_granularity_;
 }
 //Restore default partition parameters:
 partition_granularity_ = PARTITION_GRANULARITY;
 for(auto & imbalance: partition_imbalance_) imbalance = PARTITION_IMBALANCE;
 if(debugging) std::cout << "#DEBUG(ContractionSeqOptimizerMetis): The pseudo-optimal Flop count found = " << flops << std::endl;
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
 std::size_t num_miniparts = partition_granularity_;
 std::size_t contr = 0;
 bool not_done = true;
 while(not_done){
  not_done = false;
  auto num_graphs_in = graphs.size();
  for(std::size_t i = 0; i < num_graphs_in; ++i){
   auto & graph = graphs[i].first; //parent graph
   const std::size_t num_vertices = graph.getNumVertices();
   if(num_vertices > partition_max_size_){
    not_done = true;
    auto imbalance = PARTITION_IMBALANCE;
    if(contr < partition_imbalance_.size()) imbalance = partition_imbalance_[contr];
    std::size_t num_miniparts_safe = std::max(partition_factor_,std::min(num_miniparts,num_vertices/(2*partition_max_size_)));
    bool success = false;
    while(!success){
     success = graph.partitionGraph(partition_factor_,num_miniparts_safe,imbalance);
     assert(success);
     const auto num_partitions = graph.getNumPartitions(); assert(num_partitions == 2);
     for(std::size_t j = 0; j < num_partitions; ++j){
      graphs.emplace_back(std::make_pair(MetisGraph(graph,j),
                                         intermediate_num_generator())); //child graphs
      if(graphs.back().first.getNumVertices() == 0){
       if(debugging){
        std::cout << "#WARNING(ContractionSeqOptimizerMetis): Empty partition detected when splitting the parent graph with "
                  << num_vertices << " vertices into " << num_miniparts_safe << " mini-parts!" << std::endl << std::flush;
       }
       for(std::size_t k = 0; k <= j; ++k) graphs.pop_back();
       assert(num_miniparts_safe != 2);
       num_miniparts_safe = 2;
       success = false;
       break;
      }
     }
    }
    const auto last_tensor_pos = graphs.size() - 1;
    contr_seq.emplace_front(ContrTriple{graphs[i].second,
                                        graphs[last_tensor_pos-1].second,
                                        graphs[last_tensor_pos].second});
    ++contr;
   }else{
    graphs.emplace_back(graphs[i]); //parent graph is already small enough, no further division
   }
  }
  while(num_graphs_in-- > 0) graphs.pop_front(); //delete parent graphs
  num_miniparts = std::max(partition_factor_,num_miniparts/2);
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
   ++contr;
  }else if(num_vertices == 2){
   unsigned int left_id = graph.getOriginalVertexId(0);
   unsigned int right_id = graph.getOriginalVertexId(1);
   contr_seq.emplace_front(ContrTriple{graph_entry.second,left_id,right_id});
   ++contr;
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
  std::cout << "#DEBUG(ContractionSeqOptimizerMetis): Done (" << time_total.count() << " sec):";
  for(const auto & contr_pair: contr_seq) std::cout << " {" << contr_pair.left_id << ","
                                                            << contr_pair.right_id << "->"
                                                            << contr_pair.result_id << "}";
  std::cout << std::endl;
 }
 return;
}


std::unique_ptr<ContractionSeqOptimizer> ContractionSeqOptimizerMetis::createNew()
{
 return std::unique_ptr<ContractionSeqOptimizer>(new ContractionSeqOptimizerMetis());
}

} //namespace numerics

} //namespace exatn
