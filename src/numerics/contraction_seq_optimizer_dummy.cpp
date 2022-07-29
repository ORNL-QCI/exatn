/** ExaTN::Numerics: Tensor contraction sequence optimizer: Dummy
REVISION: 2019/11/08

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "contraction_seq_optimizer_dummy.hpp"
#include "tensor_network.hpp"

namespace exatn{

namespace numerics{

double ContractionSeqOptimizerDummy::determineContractionSequence(TensorNetwork & network,
                                                                  std::list<ContrTriple> & contr_seq,
                                                                  std::function<unsigned int ()> intermediate_num_generator)
{
 contr_seq.clear();
 double flops = 0.0;
 const auto num_tensors = network.getNumTensors(); //number of input tensors
 if(num_tensors > 1){
  TensorNetwork net(network);
  unsigned int ids[num_tensors], i = 0;
  for(auto iter = net.begin(); iter != net.end(); ++iter){
   if(iter->first != 0) ids[i++] = iter->first;
  }
  assert(i == num_tensors);
  unsigned int prev_tensor = ids[0];
  for(unsigned int j = 1; j < num_tensors; ++j){
   unsigned int curr_tensor = ids[j];
   if(j == (num_tensors - 1)){ //last tensor contraction
    contr_seq.emplace_back(ContrTriple{0,curr_tensor,prev_tensor});
    flops += net.getContractionCost(curr_tensor,prev_tensor);
   }else{ //intermediate tensor contraction
    auto intermediate_num = intermediate_num_generator();
    contr_seq.emplace_back(ContrTriple{intermediate_num,curr_tensor,prev_tensor});
    flops += net.getContractionCost(curr_tensor,prev_tensor);
    auto merged = net.mergeTensors(curr_tensor,prev_tensor,intermediate_num);
    assert(merged);
    prev_tensor = intermediate_num;
   }
  }
 }
 return flops;
}


std::unique_ptr<ContractionSeqOptimizer> ContractionSeqOptimizerDummy::createNew()
{
 return std::unique_ptr<ContractionSeqOptimizer>(new ContractionSeqOptimizerDummy());
}

} //namespace numerics

} //namespace exatn
