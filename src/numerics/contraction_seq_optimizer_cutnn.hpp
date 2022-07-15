/** ExaTN::Numerics: Tensor contraction sequence optimizer: cuTensorNet heuristics
REVISION: 2022/07/15

Copyright (C) 2018-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle)
Copyright (C) 2022-2022 NVIDIA Corporation

SPDX-License-Identifier: BSD-3-Clause **/

/** Rationale:
**/

#ifndef EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_CUTNN_HPP_
#define EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_CUTNN_HPP_

#ifdef CUQUANTUM

#include "contraction_seq_optimizer.hpp"

#include <memory>

#include "errors.hpp"

namespace exatn{

namespace numerics{

struct InfoCuTensorNet;


class ContractionSeqOptimizerCutnn: public ContractionSeqOptimizer{

public:

 ContractionSeqOptimizerCutnn();
 virtual ~ContractionSeqOptimizerCutnn() = default;

 void resetMinSlices(std::size_t min_slices);

 std::shared_ptr<InfoCuTensorNet> determineContractionSequenceWithSlicing(const TensorNetwork & network,
                                   std::list<ContrTriple> & contr_seq,
                                   std::function<unsigned int ()> intermediate_num_generator);

 virtual double determineContractionSequence(const TensorNetwork & network,
                                             std::list<ContrTriple> & contr_seq,
                                             std::function<unsigned int ()> intermediate_num_generator) override;

 static std::unique_ptr<ContractionSeqOptimizer> createNew();

protected:

 using ContractionSequence = std::list<ContrTriple>;

 std::size_t min_slices_;
};

} //namespace numerics

} //namespace exatn

#endif //CUQUANTUM

#endif //EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_CUTNN_HPP_
