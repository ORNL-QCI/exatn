/** ExaTN::Numerics: Tensor contraction sequence optimizer: Dummy
REVISION: 2019/09/05

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "contraction_seq_optimizer_dummy.hpp"
#include "tensor_network.hpp"

namespace exatn{

namespace numerics{

double ContractionSeqOptimizerDummy::determineContractionSequence(const TensorNetwork & network,
                                                                  std::list<ContrTriple> & contr_seq)
{
 double flops = 0.0;
 //`Finish: Requires TensorNetwork::iterator
 return flops;
}


std::unique_ptr<ContractionSeqOptimizer> ContractionSeqOptimizerDummy::createNew()
{
 return std::unique_ptr<ContractionSeqOptimizer>(new ContractionSeqOptimizerDummy());
}

} //namespace numerics

} //namespace exatn
