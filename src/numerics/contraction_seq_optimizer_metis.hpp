/** ExaTN::Numerics: Tensor contraction sequence optimizer: Metis heuristics
REVISION: 2020/05/20

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
**/

#ifndef EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_METIS_HPP_
#define EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_METIS_HPP_

#include "contraction_seq_optimizer.hpp"

#include <vector>

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

 using ContractionSequence = std::list<ContrTriple>;

 void determineContrSequence(const TensorNetwork & network,
                             std::list<ContrTriple> & contr_seq,
                             std::function<unsigned int ()> intermediate_num_generator);

 static constexpr const unsigned int NUM_WALKERS = 32;
 static constexpr const double ACCEPTANCE_TOLERANCE = 0.0;

 static constexpr const std::size_t PARTITION_FACTOR = 2;
 static constexpr const std::size_t PARTITION_MAX_SIZE = 3;
 static constexpr const std::size_t PARTITION_IMBALANCE_DEPTH = 48;
 static constexpr const std::size_t PARTITION_GRANULARITY = PARTITION_IMBALANCE_DEPTH;
 static constexpr const double PARTITION_IMBALANCE = 1.3;

 unsigned int num_walkers_;
 double acceptance_tolerance_;

 std::size_t partition_factor_;
 std::size_t partition_granularity_;
 std::size_t partition_max_size_;
 std::vector<double> partition_imbalance_;
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_METIS_HPP_
