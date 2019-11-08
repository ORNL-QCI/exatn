/** ExaTN::Numerics: Tensor contraction sequence optimizer: Dummy
REVISION: 2019/11/08

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
**/

#ifndef EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_DUMMY_HPP_
#define EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_DUMMY_HPP_

#include "contraction_seq_optimizer.hpp"

namespace exatn{

namespace numerics{

class ContractionSeqOptimizerDummy: public ContractionSeqOptimizer{

public:

 virtual double determineContractionSequence(const TensorNetwork & network,
                                             std::list<ContrTriple> & contr_seq,
                                             std::function<unsigned int ()> intermediate_num_generator) override;

 static std::unique_ptr<ContractionSeqOptimizer> createNew();
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_DUMMY_HPP_
