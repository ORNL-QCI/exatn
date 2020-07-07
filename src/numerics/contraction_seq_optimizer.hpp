/** ExaTN::Numerics: Tensor contraction sequence optimizer
REVISION: 2020/07/06

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
**/

#ifndef EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_HPP_
#define EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_HPP_

#include "tensor_basic.hpp"

#include <list>
#include <memory>
#include <unordered_map>
#include <functional>

namespace exatn{

namespace numerics{

//Tensor contraction triple:
struct ContrTriple{
 unsigned int result_id; //id of the tensor-result (new)
 unsigned int left_id;   //id of the left input tensor (old)
 unsigned int right_id;  //id of the right input tensor (old)
};

class TensorNetwork;
class MetisGraph;


class ContractionSeqOptimizer{

public:

 virtual ~ContractionSeqOptimizer() = default;

 /** Determines the pseudo-optimal tensor contraction sequence required for
     evaluating a given tensor network. The unique intermediate tensor id's are generated
     by the provided intermediate number generator (each invocation returns a new tensor id).
     The latter can be conveniently passed as a lambda closure. The returned double value
     is an estimate of the total flop count associated with the determined contraction sequence.
     The tensor network must have at least two input tensors in order to get a single contraction.
     No contraction sequence is generated for tensor networks with a single input tensor. **/
 virtual double determineContractionSequence(const TensorNetwork & network,
                                             std::list<ContrTriple> & contr_seq,
                                             std::function<unsigned int ()> intermediate_num_generator) = 0;

 /** Caches the determined pseudo-optimal tensor contraction sequence for a given
     tensor network for a later retrieval for the same tensor networks. Returns TRUE
     on success, FALSE in case this tensor network has already been cached before. **/
 static bool cacheContractionSequence(const TensorNetwork & network); //in: tensor network with a determined tensor contraction sequence

 /** Erases the previously cached tensor contraction sequence for a given tensor
     network and returns TRUE, or returns FALSE in case it has not been cached before. **/
 static bool eraseContractionSequence(const TensorNetwork & network); //in: tensor network

 /** Retrieves a previously cached tensor contraction sequence for a given tensor
     network. Returns nullptr in case it has not been cached before. **/
 static const std::list<ContrTriple> * findContractionSequence(const TensorNetwork & network); //in: tensor network

private:

 /** Cached tensor contraction sequences. **/
 static std::unordered_map<std::string,                     //tensor network name
                           std::pair<MetisGraph,            //METIS graph of the tensor network
                                     std::list<ContrTriple> //tensor contraction sequence
                                    >> cached_contr_seqs_;
};

using createContractionSeqOptimizerFn = std::unique_ptr<ContractionSeqOptimizer> (*)(void);

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_HPP_
