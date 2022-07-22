/** ExaTN::Numerics: Tensor contraction sequence optimizer: cuTensorNet heuristics
REVISION: 2022/07/22

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
#include "mpi_proxy.hpp"

#include <memory>

#include "errors.hpp"

namespace exatn{

namespace numerics{

struct InfoCuTensorNet;


class ContractionSeqOptimizerCutnn: public ContractionSeqOptimizer{

public:

 ContractionSeqOptimizerCutnn();
 virtual ~ContractionSeqOptimizerCutnn();

 virtual void resetMemLimit(std::size_t mem_limit) override;

 virtual void resetMinSlices(std::size_t min_slices) override;

 virtual double determineContractionSequence(const TensorNetwork & network,
                                             std::list<ContrTriple> & contr_seq,
                                             std::function<unsigned int ()> intermediate_num_generator) override;

 static std::unique_ptr<ContractionSeqOptimizer> createNew();

 /** Returns information on index splitting done by cuTensorNet: {{tensor_id,index_position},segment_size}.
     The tensor id and index position inside the tensor are returned for the first encounter of that index. **/
 static std::vector<std::pair<std::pair<unsigned int, unsigned int>, DimExtent>> extractIndexSplittingInfo(const TensorNetwork & network);

protected:

 using ContractionSequence = std::list<ContrTriple>;

 std::shared_ptr<InfoCuTensorNet> determineContractionSequenceWithSlicing(
                                   const TensorNetwork & network,
                                   std::list<ContrTriple> & contr_seq,
                                   std::function<unsigned int ()> intermediate_num_generator);

 std::size_t mem_limit_; //memory limit (for intermediates)
 std::size_t min_slices_; //min number of slices to produce
 void * cutnn_handle_; //cutensornet handle
};

} //namespace numerics

} //namespace exatn

#endif //CUQUANTUM

#endif //EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_CUTNN_HPP_
