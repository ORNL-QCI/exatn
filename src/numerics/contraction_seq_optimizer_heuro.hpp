/** ExaTN::Numerics: Tensor contraction sequence optimizer: Greedy heuristics
REVISION: 2020/04/28

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Greedy heuristics based on the individual tensor contraction cost.
**/

#ifndef EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_HEURO_HPP_
#define EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_HEURO_HPP_

#include "contraction_seq_optimizer.hpp"

#include "errors.hpp"

namespace exatn{

namespace numerics{

class ContractionSeqOptimizerHeuro: public ContractionSeqOptimizer{

public:

 ContractionSeqOptimizerHeuro();
 virtual ~ContractionSeqOptimizerHeuro() = default;

 void resetNumWalkers(unsigned int num_walkers);

 virtual double determineContractionSequence(TensorNetwork & network,
                                             std::list<ContrTriple> & contr_seq,
                                             std::function<unsigned int ()> intermediate_num_generator) override;

 static std::unique_ptr<ContractionSeqOptimizer> createNew();

protected:

 static constexpr const unsigned int NUM_WALKERS = 8;

 unsigned int num_walkers_;
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_HEURO_HPP_
