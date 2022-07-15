/** ExaTN::Numerics: Tensor contraction sequence optimizer factory
REVISION: 2022/07/14

Copyright (C) 2018-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle)
Copyright (C) 2022-2022 NVIDIA Corporation

SPDX-License-Identifier: BSD-3-Clause **/

/** Rationale:
 (a) Creates tensor contraction sequence optimizers of desired kind.
**/

#ifndef EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_FACTORY_HPP_
#define EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_FACTORY_HPP_

#include "tensor_basic.hpp"
#include "contraction_seq_optimizer.hpp"
#include "contraction_seq_optimizer_dummy.hpp"
#include "contraction_seq_optimizer_heuro.hpp"
#include "contraction_seq_optimizer_greed.hpp"
#include "contraction_seq_optimizer_metis.hpp"
#ifdef CUQUANTUM
#include "contraction_seq_optimizer_cutnn.hpp"
#endif

#include <string>
#include <memory>
#include <map>

#include "errors.hpp"

namespace exatn{

namespace numerics{

class ContractionSeqOptimizerFactory{
public:

 ContractionSeqOptimizerFactory(const ContractionSeqOptimizerFactory &) = delete;
 ContractionSeqOptimizerFactory & operator=(const ContractionSeqOptimizerFactory &) = delete;
 ContractionSeqOptimizerFactory(ContractionSeqOptimizerFactory &&) noexcept = default;
 ContractionSeqOptimizerFactory & operator=(ContractionSeqOptimizerFactory &&) noexcept = default;
 ~ContractionSeqOptimizerFactory() = default;

 /** Registers a new tensor contraction optimizer subtype to produce instances of. **/
 void registerContractionSeqOptimizer(const std::string & name, createContractionSeqOptimizerFn creator);

 /** Creates a new instance of a desired subtype. **/
 std::unique_ptr<ContractionSeqOptimizer> createContractionSeqOptimizer(const std::string & name);
 /** Creates a new instance of a desired subtype. **/
 std::shared_ptr<ContractionSeqOptimizer> createContractionSeqOptimizerShared(const std::string & name);

 /** Returns a pointer to the ContractionSeqOptimizerFactory singleton. **/
 static ContractionSeqOptimizerFactory * get();

private:

 ContractionSeqOptimizerFactory(); //private ctor

 std::map<std::string,createContractionSeqOptimizerFn> factory_map_;

};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_FACTORY_HPP_
