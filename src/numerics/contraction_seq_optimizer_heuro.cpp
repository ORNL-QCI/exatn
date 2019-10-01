/** ExaTN::Numerics: Tensor contraction sequence optimizer: Heuristics
REVISION: 2019/10/01

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "contraction_seq_optimizer_heuro.hpp"
#include "tensor_network.hpp"

#include <cassert>

#include <vector>
#include <queue>
#include <tuple>
#include <iterator>
#include <chrono>

namespace exatn{

namespace numerics{

static constexpr unsigned int NUM_WALKERS = 1024; //default number of walkers for tensor contraction sequence optimization


ContractionSeqOptimizerHeuro::ContractionSeqOptimizerHeuro():
 num_walkers_(NUM_WALKERS)
{
}


void ContractionSeqOptimizerHeuro::resetNumWalkers(unsigned int num_walkers)
{
 num_walkers_ = num_walkers;
 return;
}


double ContractionSeqOptimizerHeuro::determineContractionSequence(const TensorNetwork & network,
                                                                  std::list<ContrTriple> & contr_seq,
                                                                  std::function<unsigned int ()> intermediate_num_generator)
{
 using ContractionSequence = std::list<ContrTriple>;
 using ContrPath = std::tuple<TensorNetwork, ContractionSequence, double>;

 contr_seq.clear();
 double flops = 0.0;

 auto numContractions = network.getNumTensors() - 1; //number of contractions is one less than the number of r.h.s. tensors
 if(numContractions == 0) return flops;

 std::cout << "#DEBUG(ContractionSeqOptimizerHeuro): Determining a pseudo-optimal tensor contraction sequence ... "; //debug
 auto timeBeg = std::chrono::high_resolution_clock::now();

 ContractionSequence contrSeqEmpty;
 std::vector<ContrPath> inputPaths; //input: vector
 inputPaths.emplace_back(std::make_tuple(network,contrSeqEmpty,0.0)); //initial configuration

 auto cmpPaths = [](const ContrPath & left, const ContrPath & right){return (std::get<2>(left) < std::get<2>(right));};
 std::priority_queue<ContrPath, std::vector<ContrPath>, decltype(cmpPaths)> priq(cmpPaths); //output: priority queue

 for(decltype(numContractions) pass = 0; pass < numContractions; ++pass){
  unsigned int numPassCands = 0;
  for(auto & contrPath: inputPaths){
   auto & parentTensNet = std::get<0>(contrPath); //parental tensor network
   const auto numTensors = parentTensNet.getNumTensors(); //number of r.h.s. tensors in the parental tensor network
   const auto & parentContrSeq = std::get<1>(contrPath); //parental contraction sequence
   for(auto iter_i = parentTensNet.begin(); iter_i != parentTensNet.end(); ++iter_i){ //r.h.s. tensors
    auto i = iter_i->first;
    for(auto iter_j = std::next(iter_i); iter_j != parentTensNet.end(); ++iter_j){ //r.h.s. tensors
     auto j = iter_j->first;
     double contrCost = parentTensNet.getContractionCost(i,j); //tensor contraction cost
     //std::cout << std::endl << "New candidate contracted pair of tensors is {" << i << "," << j << "} with cost " << contrCost; //debug
     TensorNetwork tensNet(parentTensNet);
     auto contracted = tensNet.mergeTensors(i,j,0); assert(contracted);
     auto cSeq = parentContrSeq;
     cSeq.emplace_back(ContrTriple{0,i,j}); //append a new pair of contracted tensors
     priq.emplace(std::make_tuple(tensNet, cSeq, contrCost + std::get<2>(contrPath))); //cloning tensor network and contraction sequence
     if(priq.size() > num_walkers_) priq.pop();
     numPassCands++;
    }
   }
  }
  std::cout << std::endl << "Pass " << pass << ": Total number of candidates considered = " << numPassCands; //debug
  inputPaths.clear();
  if(pass == numContractions - 1){ //last pass
   while(priq.size() > 1) priq.pop();
   contr_seq = std::get<1>(priq.top());
   std::cout << std::endl << "Best tensor contraction sequence cost found = " << std::get<2>(priq.top()); //debug
   priq.pop();
  }else{
   while(priq.size() > 0){
    inputPaths.emplace_back(priq.top());
    priq.pop();
   }
  }
 }

 auto timeEnd = std::chrono::high_resolution_clock::now();
 auto timeTot = std::chrono::duration_cast<std::chrono::duration<double>>(timeEnd - timeBeg);
 std::cout << std::endl << "Done (" << timeTot.count() << " sec):"; //debug
 for(const auto & cPair: contr_seq) std::cout << " {" << cPair.left_id << "," << cPair.right_id << "}"; //debug
 std::cout << std::endl; //debug

 return flops;
}


std::unique_ptr<ContractionSeqOptimizer> ContractionSeqOptimizerHeuro::createNew()
{
 return std::unique_ptr<ContractionSeqOptimizer>(new ContractionSeqOptimizerHeuro());
}

} //namespace numerics

} //namespace exatn
