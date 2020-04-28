/** ExaTN::Numerics: Tensor contraction sequence optimizer: Metis heuristics
REVISION: 2020/04/28

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
**/

#ifndef EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_METIS_HPP_
#define EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_METIS_HPP_

#include "contraction_seq_optimizer.hpp"

namespace exatn{

namespace numerics{

class ContractionSeqOptimizerMetis: public ContractionSeqOptimizer{

public:

 ContractionSeqOptimizerMetis();
 virtual ~ContractionSeqOptimizerMetis() = default;

 void resetNumWalkers(unsigned int num_walkers);

 void resetAcceptanceTolerance(double acceptance_tolerance);

 virtual double determineContractionSequence(const TensorNetwork & network,
                                             std::list<ContrTriple> & contr_seq,
                                             std::function<unsigned int ()> intermediate_num_generator) override;

 static std::unique_ptr<ContractionSeqOptimizer> createNew();

protected:

 static const unsigned int NUM_WALKERS = 1;

 unsigned int num_walkers_;
 double acceptance_tolerance_;
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_METIS_HPP_
