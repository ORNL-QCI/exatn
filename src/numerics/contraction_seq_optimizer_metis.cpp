/** ExaTN::Numerics: Tensor contraction sequence optimizer: Metis heuristics
REVISION: 2020/04/28

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

 using ContractionSequence = std::list<ContrTriple>;

 contr_seq.clear();
 double flops = 0.0;

 auto numContractions = network.getNumTensors() - 1; //number of contractions is one less than the number of r.h.s. tensors
 if(numContractions == 0) return flops;

 if(debugging) std::cout << "#DEBUG(ContractionSeqOptimizerMetis): Determining a pseudo-optimal tensor contraction sequence ... \n"; //debug
 auto timeBeg = std::chrono::high_resolution_clock::now();

 //Recursive Kway partitioning:
 std::deque<MetisGraph> graphs;
 graphs.emplace_back(MetisGraph(network));
 bool not_done = true;
 while(not_done){
  not_done = false;
  auto num_graphs_in = graphs.size();
  for(std::size_t i = 0; i < num_graphs_in; ++i){
   auto & graph = graphs[i];
   if(graph.getNumVertices() > partition_max_size_){
    not_done = true;
    bool success = graph.partitionGraph(partition_factor_,partition_imbalance_); assert(success);
    const auto num_partitions = graph.getNumPartitions(); assert(num_partitions > 1);
    for(std::size_t j = 0; j < num_partitions; ++j) graphs.emplace_back(MetisGraph(graph,j));
   }else{
    graphs.emplace_back(graph);
   }
  }
  while(num_graphs_in-- > 0) graphs.pop_front();
 }
 //Assemble the tensor contraction sequence:
 

 auto timeEnd = std::chrono::high_resolution_clock::now();
 auto timeTot = std::chrono::duration_cast<std::chrono::duration<double>>(timeEnd - timeBeg);
 if(debugging){
  std::cout << "#DEBUG(ContractionSeqOptimizerMetis): Done (" << timeTot.count() << " sec):"; //debug
  for(const auto & cPair: contr_seq) std::cout << " {" << cPair.left_id << "," << cPair.right_id
                                               << "->" << cPair.result_id <<"}"; //debug
  std::cout << std::endl; //debug
 }
 return flops;
}


std::unique_ptr<ContractionSeqOptimizer> ContractionSeqOptimizerMetis::createNew()
{
 return std::unique_ptr<ContractionSeqOptimizer>(new ContractionSeqOptimizerMetis());
}

} //namespace numerics

} //namespace exatn
