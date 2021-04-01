/** ExaTN::Numerics: Tensor contraction sequence optimizer
REVISION: 2021/04/01

Copyright (C) 2018-2021 Thien Ngyuen, Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
**/

#ifndef EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_HPP_
#define EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_HPP_

#include "tensor_basic.hpp"

#include <list>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>

#include "errors.hpp"

namespace exatn{

namespace numerics{

//Tensor contraction triple:
struct ContrTriple{
 unsigned int result_id; //id of the tensor-result (new)
 unsigned int left_id;   //id of the left input tensor (old)
 unsigned int right_id;  //id of the right input tensor (old)
};

// Tensor Slice Info
struct SliceIndex {
 unsigned int tensor_id; // id of the tensor
 unsigned int leg_id;    // leg id of the tensor which we should slice.
};

class TensorNetwork;
class MetisGraph;

//Free functions:
void packContractionSequenceIntoVector(const std::list<ContrTriple> & contr_sequence,
                                       std::vector<unsigned int> & contr_sequence_content);
void unpackContractionSequenceFromVector(std::list<ContrTriple> & contr_sequence,
                                         const std::vector<unsigned int> & contr_sequence_content);


class ContractionSeqOptimizer{

public:

 virtual ~ContractionSeqOptimizer() = default;

 /** Determines the pseudo-optimal tensor contraction sequence required for
     evaluating a given tensor network. The unique intermediate tensor id's are generated
     by the provided intermediate number generator (each invocation returns a new tensor id).
     The latter can be conveniently passed as a lambda closure. The returned double value
     is an estimate of the total FMA flop count associated with the determined contraction sequence.
     The tensor network must have at least two input tensors in order to get a single contraction.
     No contraction sequence is generated for tensor networks with a single input tensor. Note that
     the FMA flop count neither includes the FMA factor of 2.0 nor the factor of 4.0 for complex numbers. **/
 virtual double determineContractionSequence(const TensorNetwork & network,
                                             std::list<ContrTriple> & contr_seq,
                                             std::function<unsigned int ()> intermediate_num_generator) = 0;

 /** Determines the pseudo-optimal tensor contraction sequence required for
     evaluating a given tensor network:
     - target_slice_size (in bytes): slice until largest tensor is at maximum this size;
     - slice_inds: list of slice indices (tensor Id and leg Id pairs).
     If the ContractionSeqOptimizer sub-class doesn't explicitly implement this
     method, will just fallback to the above non-slicing method. **/
 virtual double determineContractionSequence(const TensorNetwork & network,
                                             std::list<ContrTriple> & contr_seq,
                                             std::function<unsigned int()> intermediate_num_generator,
                                             uint64_t target_slice_size,
                                             std::list<SliceIndex> &slice_inds) {
  return determineContractionSequence(network, contr_seq, intermediate_num_generator);
 }

 /** Caches the determined pseudo-optimal tensor contraction sequence for a given
     tensor network for a later retrieval for the same tensor networks. Returns TRUE
     on success, FALSE in case this tensor network has already been cached before. **/
 static bool cacheContractionSequence(const TensorNetwork & network); //in: tensor network with a determined tensor contraction sequence

 /** Erases the previously cached tensor contraction sequence for a given tensor
     network and returns TRUE, or returns FALSE in case it has not been cached before. **/
 static bool eraseContractionSequence(const TensorNetwork & network); //in: tensor network

 /** Retrieves a previously cached tensor contraction sequence for a given tensor
     network and its corresponding FMA flop count. Returns {nullptr,0.0} in case
     no previously cached tensor contraction sequence has been found. **/
 static std::pair<const std::list<ContrTriple> *, double> findContractionSequence(const TensorNetwork & network); //in: tensor network

private:

 //Cached optimized tensor contraction sequence:
 struct CachedContrSeq{
  std::shared_ptr<MetisGraph> graph; //METIS graph of the tensor network
  std::list<ContrTriple> contr_seq;  //optimized tensor contraction sequence for the tensor network
  double fma_flops;                  //FMA flop count for the stored tensor contraction sequence
 };

 /** Cached tensor contraction sequences. **/
 static std::unordered_map<std::string,CachedContrSeq> cached_contr_seqs_; //tensor network name --> optimized tensor contraction sequence
};

using createContractionSeqOptimizerFn = std::unique_ptr<ContractionSeqOptimizer> (*)(void);

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_CONTRACTION_SEQ_OPTIMIZER_HPP_
