/** ExaTN::Numerics: Tensor contraction sequence optimizer: CuTensorNet heuristics
REVISION: 2022/07/15

Copyright (C) 2018-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle)
Copyright (C) 2022-2022 NVIDIA Corporation

SPDX-License-Identifier: BSD-3-Clause **/

#ifdef CUQUANTUM

#include "tensor_network.hpp"
#include "contraction_seq_optimizer_cutnn.hpp"

#include <cutensornet.h>
#include <cutensor.h>
#include <cuda_runtime.h>

namespace exatn{

namespace numerics{

struct InfoCuTensorNet{
 cutensornetNetworkDescriptor_t cutnn_network;
 cutensornetContractionOptimizerConfig_t cutnn_config;
 cutensornetContractionOptimizerInfo_t cutnn_info;
};


ContractionSeqOptimizerCutnn::ContractionSeqOptimizerCutnn():
 min_slices_(1)
{
}


void ContractionSeqOptimizerCutnn::resetMinSlices(std::size_t min_slices)
{
 make_sure(min_slices > 0,"#ERROR(exatn::numerics::ContractionSeqOptimizerCutnn): Minimal number of slices must be greater than zero!");
 min_slices_ = min_slices;
 return;
}


std::shared_ptr<InfoCuTensorNet> ContractionSeqOptimizerCutnn::determineContractionSequenceWithSlicing(
                                  const TensorNetwork & network,
                                  std::list<ContrTriple> & contr_seq,
                                  std::function<unsigned int ()> intermediate_num_generator)
{
 //`Implement
 return std::shared_ptr<InfoCuTensorNet>{nullptr};
}


double ContractionSeqOptimizerCutnn::determineContractionSequence(const TensorNetwork & network,
                                                                  std::list<ContrTriple> & contr_seq,
                                                                  std::function<unsigned int ()> intermediate_num_generator)
{
 double flops = 0.0;
 //`Implement
 return flops;
}


std::unique_ptr<ContractionSeqOptimizer> ContractionSeqOptimizerCutnn::createNew()
{
 return std::unique_ptr<ContractionSeqOptimizer>(new ContractionSeqOptimizerCutnn());
}

} //namespace numerics

} //namespace exatn

#endif //CUQUANTUM
