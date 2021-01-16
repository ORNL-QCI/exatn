/** ExaTN:: Variational optimizer of a closed symmetric tensor network expansion functional
REVISION: 2021/01/16

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

#include "optimizer.hpp"

namespace exatn{

unsigned int TensorNetworkOptimizer::debug{0};


TensorNetworkOptimizer::TensorNetworkOptimizer(std::shared_ptr<TensorOperator> tensor_operator,
                                               std::shared_ptr<TensorExpansion> vector_expansion,
                                               double tolerance):
 tensor_operator_(tensor_operator), vector_expansion_(vector_expansion),
 max_iterations_(DEFAULT_MAX_ITERATIONS), epsilon_(DEFAULT_LEARN_RATE), tolerance_(tolerance)
{
}


void TensorNetworkOptimizer::resetTolerance(double tolerance)
{
 tolerance_ = tolerance;
 return;
}


void TensorNetworkOptimizer::resetLearningRate(double learn_rate)
{
 epsilon_ = learn_rate;
 return;
}


void TensorNetworkOptimizer::resetMaxIterations(unsigned int max_iterations)
{
 max_iterations_ = max_iterations;
 return;
}


std::shared_ptr<TensorExpansion> TensorNetworkOptimizer::getSolution() const
{
 return vector_expansion_;
}


bool TensorNetworkOptimizer::optimize()
{
 return optimize(exatn::getDefaultProcessGroup());
}

bool TensorNetworkOptimizer::optimize(const ProcessGroup & process_group)
{
 unsigned int local_rank; //local process rank within the process group
 if(!process_group.rankIsIn(exatn::getProcessRank(),&local_rank)) return true; //process is not in the group: Do nothing

 bool success = true;
 //`Finish
 return success;
}


void TensorNetworkOptimizer::resetDebugLevel(unsigned int level)
{
 TensorNetworkOptimizer::debug = level;
 return;
}

} //namespace exatn
