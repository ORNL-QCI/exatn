/** ExaTN::Numerics: Tensor contraction sequence optimizer: Heuristics
REVISION: 2019/09/09

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "contraction_seq_optimizer_heuro.hpp"
#include "tensor_network.hpp"

namespace exatn{

namespace numerics{

static constexpr unsigned int NUM_WALKERS = 1024; //default number of walkers for tensor contraction sequence optimization


double ContractionSeqOptimizerHeuro::determineContractionSequence(const TensorNetwork & network,
                                                                  std::list<ContrTriple> & contr_seq,
                                                                  unsigned int intermediate_num_begin)
{
 contr_seq.clear();
 double flops = 0.0;
 //`Finish
 return flops;
}


std::unique_ptr<ContractionSeqOptimizer> ContractionSeqOptimizerHeuro::createNew()
{
 return std::unique_ptr<ContractionSeqOptimizer>(new ContractionSeqOptimizerHeuro());
}

} //namespace numerics

} //namespace exatn
