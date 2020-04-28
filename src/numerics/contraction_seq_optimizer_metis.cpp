/** ExaTN::Numerics: Tensor contraction sequence optimizer: Metis heuristics
REVISION: 2020/04/28

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "contraction_seq_optimizer_metis.hpp"
#include "tensor_network.hpp"

#include <cassert>

#include <vector>
#include <queue>
#include <tuple>
#include <iterator>
#include <chrono>

namespace exatn{

namespace numerics{

ContractionSeqOptimizerMetis::ContractionSeqOptimizerMetis():
 num_walkers_(NUM_WALKERS)
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
 constexpr bool debugging = false;

 using ContractionSequence = std::list<ContrTriple>;
 using ContrPath = std::tuple<TensorNetwork,       //0: current state of the tensor network
                              ContractionSequence, //1: tensor contraction sequence resulted in this state
                              double,              //2: current total flop count
                              double>;             //3: local differential volume (temporary)

 contr_seq.clear();
 double flops = 0.0;

 auto numContractions = network.getNumTensors() - 1; //number of contractions is one less than the number of r.h.s. tensors
 if(numContractions == 0) return flops;

 if(debugging) std::cout << "#DEBUG(ContractionSeqOptimizerMetis): Determining a pseudo-optimal tensor contraction sequence ... \n"; //debug
 auto timeBeg = std::chrono::high_resolution_clock::now();

 ContractionSequence contrSeqEmpty;
 std::vector<ContrPath> inputPaths; //considered contraction paths
 inputPaths.emplace_back(std::make_tuple(network,contrSeqEmpty,0.0,0.0)); //initial configuration

 //auto cmpPaths = [](const ContrPath & left, const ContrPath & right){return (std::get<3>(left) < std::get<3>(right));};
 auto cmpPaths = [](const ContrPath & left, const ContrPath & right){
                    if(std::get<3>(left) == std::get<3>(right)) return (std::get<2>(left) < std::get<2>(right));
                    return (std::get<3>(left) < std::get<3>(right));
                   };
 std::priority_queue<ContrPath, std::vector<ContrPath>, decltype(cmpPaths)> priq(cmpPaths); //prioritized contraction paths

 //Loop over the tensor contractions (passes):
 for(decltype(numContractions) pass = 0; pass < numContractions; ++pass){
  if(debugging){
   std::cout << "#DEBUG(ContractionSeqOptimizerMetis): Pass " << pass << " started with "
             << inputPaths.size() << " candidates" << std::endl; //debug
  }
  unsigned int intermediate_id = intermediate_num_generator(); //id of the next intermediate tensor
  unsigned int numPassCands = 0, candid = 0;
  //Update the list of promising contraction path candidates due to a new tensor contraction:
  for(auto & contrPath: inputPaths){
   //std::cout << " #DEBUG(ContractionSeqOptimizerMetis): Processing candidate path " << candid++ << std::endl; //debug
   auto & parentTensNet = std::get<0>(contrPath); //parental tensor network
   const auto numTensors = parentTensNet.getNumTensors(); //number of r.h.s. tensors in the parental tensor network
   const auto & parentContrSeq = std::get<1>(contrPath); //contraction sequence for the parental tensor network
   //Inspect contractions of all unique pairs of tensors:
   for(auto iter_i = parentTensNet.begin(); iter_i != parentTensNet.end(); ++iter_i){ //r.h.s. tensors
    auto i = iter_i->first;
    if(i != 0){ //exclude output tensor
     const auto & tensor_i = iter_i->second; //connected tensor i
     //std::cout << "  First tensor id " << i << std::endl; //debug
     const auto adjacent_tensors = parentTensNet.getAdjacentTensors(i);
     if(!adjacent_tensors.empty()){
      for(auto & j: adjacent_tensors){
       if(j > i){ //unique pairs
        const auto & tensor_j = *(parentTensNet.getTensorConn(j));
        double diff_vol;
        double contrCost = getTensorContractionCost(tensor_i,tensor_j,&diff_vol); //tensor contraction cost (flops)
        //double contrCost = parentTensNet.getContractionCost(i,j,&diff_vol); //tensor contraction cost (flops)
        //std::cout << "  New candidate contracted pair of tensors is {" << i << "," << j << "} with cost " << contrCost << std::endl; //debug
        TensorNetwork tensNet(parentTensNet);
        auto contracted = tensNet.mergeTensors(i,j,intermediate_id); assert(contracted);
        auto cSeq = parentContrSeq;
        if(pass == numContractions - 1){ //the very last tensor contraction writes into the output tensor #0
         cSeq.emplace_back(ContrTriple{0,i,j}); //append the last pair of contracted tensors
        }else{
         cSeq.emplace_back(ContrTriple{intermediate_id,i,j}); //append a new pair of contracted tensors
        }
        priq.emplace(std::make_tuple(tensNet, cSeq, contrCost + std::get<2>(contrPath), diff_vol)); //cloning tensor network and contraction sequence
        if(priq.size() > num_walkers_) priq.pop(); //remove the top-costly contraction path when limit achieved
        numPassCands++;
       }
      }
     }else{
      for(auto iter_j = std::next(iter_i); iter_j != parentTensNet.end(); ++iter_j){ //r.h.s. tensors
       auto j = iter_j->first;
       if(j != 0){ //exclude output tensor
        const auto & tensor_j = iter_j->second; //connected tensor j
        double diff_vol;
        double contrCost = getTensorContractionCost(tensor_i,tensor_j,&diff_vol); //tensor contraction cost (flops)
        //double contrCost = parentTensNet.getContractionCost(i,j,&diff_vol); //tensor contraction cost (flops)
        //std::cout << "  New candidate contracted pair of tensors is {" << i << "," << j << "} with cost " << contrCost << std::endl; //debug
        TensorNetwork tensNet(parentTensNet);
        auto contracted = tensNet.mergeTensors(i,j,intermediate_id); assert(contracted);
        auto cSeq = parentContrSeq;
        if(pass == numContractions - 1){ //the very last tensor contraction writes into the output tensor #0
         cSeq.emplace_back(ContrTriple{0,i,j}); //append the last pair of contracted tensors
        }else{
         cSeq.emplace_back(ContrTriple{intermediate_id,i,j}); //append a new pair of contracted tensors
        }
        priq.emplace(std::make_tuple(tensNet, cSeq, contrCost + std::get<2>(contrPath), diff_vol)); //cloning tensor network and contraction sequence
        if(priq.size() > num_walkers_) priq.pop(); //remove the top-costly contraction path when limit achieved
        numPassCands++;
       }
      }
     }
    }
   }
  }
  if(debugging){
   std::cout << "#DEBUG(ContractionSeqOptimizerMetis): Pass " << pass << ": Total number of candidates considered = "
             << numPassCands << std::endl; //debug
  }
  //Collect the cheapest contraction paths left:
  inputPaths.clear();
  if(pass == numContractions - 1){ //last pass
   while(priq.size() > 1) priq.pop(); //get to the cheapest contraction path
   contr_seq = std::get<1>(priq.top());
   flops = std::get<2>(priq.top());
   priq.pop();
   if(debugging){
    std::cout << "#DEBUG(ContractionSeqOptimizerMetis): Best tensor contraction sequence found has cost (flops) = "
              << flops << std::endl; //debug
   }
  }else{ //intermediate pass
   while(priq.size() > 0){
    inputPaths.emplace_back(priq.top());
    priq.pop();
   }
  }
 }

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
