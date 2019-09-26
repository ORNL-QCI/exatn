/** ExaTN::Numerics: Tensor network
REVISION: 2019/09/26

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) A tensor network is a set of connected tensors.
     Each tensor in a tensor network can be connected to
     other tensors in that tensor network via tensor legs.
     Each tensor leg in a given tensor is uniquely associated
     with one of its modes, one tensor leg per tensor mode. The
     numeration of tensor modes is contiguous and starts from 0.
     A tensor leg can connect a given tensor with one or more
     other tensors in the same tensor network. Thus, tensor legs
     can be binary, ternary, etc., in general (binary is common choice).
 (b) A tensor network is always closed, which requires introducing
     an explicit output tensor collecting all open legs of the original
     tensor network. If the original tensor network does not have open
     legs, the output tensor is simply a scalar which the original tensor
     network evaluates to; otherwise, a tensor network evaluates to a tensor.
 (c) Current tensor enumeration (it is just one option):
       0: Output tensor/scalar which the tensor network evaluates to;
       1..N: Input tensors/scalars constituting the original tensor network;
       N+1..M: Intermediate tensors obtained by contractions of the input tensors.
     In general, only the output tensor is required to have id = 0; any other
     tensor in the tensor network may have any unique positive id.
 (d) Building a tensor network:
     Option 1: A new tensor can be appended into a tensor network by either:
               (1) Explicitly matching the tensor modes with the modes of all
                   other tensors present or to be present in the tensor network.
                   The fully specified output tensor with all its legs has had to
                   be provided in advance in the TensorNetwork ctor. This way
                   requires the advance knowledge of the entire tensor network.
                   Once all tensors have been appended, one needs to call .finalize()
                   to complete the construction of the tensor network.
               (2) Matching the tensor modes with the modes of the current output
                   tensor of the tensor network. In this case, the unmatched modes
                   of the newly appended tensor will be appended to the current
                   output tensor of the tensor network (at the end).
     Option 2: A tensor network can be appended to another tensor network by
               matching the modes of the output tensors of both tensor networks.
               The unmatched modes of the output tensor of the appended tensor
               network will be appended to the output tensor of the primary
               tensor network (at the end). The appended tensor network will
               cease to exist after being absorbed by the primary tensor network.
 (e) The modes of the output tensor of a tensor network can be examined and reordered.
 (f) Any tensor except the output tensor can be deleted from the tensor network.
 (g) Any two tensors, excluding the output tensor, can be merged by tensor contraction.
**/

#ifndef EXATN_NUMERICS_TENSOR_NETWORK_HPP_
#define EXATN_NUMERICS_TENSOR_NETWORK_HPP_

#include "tensor_basic.hpp"
#include "tensor_connected.hpp"
#include "tensor_op_factory.hpp"
#include "network_build_factory.hpp"
#include "contraction_seq_optimizer.hpp"

#include <unordered_map>
#include <map>
#include <vector>
#include <list>
#include <string>
#include <memory>

namespace exatn{

namespace numerics{

class TensorNetwork{
public:

 using Iterator = typename std::unordered_map<unsigned int, TensorConn>::iterator; //iterator
 using ConstIterator = typename std::unordered_map<unsigned int, TensorConn>::const_iterator; //constant iterator

 /** Creates an unnamed empty tensor network with a single scalar output tensor named "_SMOKY_TENSOR_" **/
 TensorNetwork();
 /** Creates a named empty tensor network with a single scalar output tensor named with the same name. **/
 TensorNetwork(const std::string & name);
 /** Creates a named empty tensor network with an explicitly provided output tensor with the same name. **/
 TensorNetwork(const std::string & name,                    //in: tensor network name
               std::shared_ptr<Tensor> output_tensor,       //in: fully specified output tensor of the tensor network
               const std::vector<TensorLeg> & output_legs); //in: fully specified output tensor legs
 /** Creates a named tensor network from a symbolic tensor network expression and a container of tensors. **/
 TensorNetwork(const std::string & name,                    //in: tensor network name
               const std::string & tensor_network,          //in: tensor network expression (symbolic math expression)
               const std::map<std::string,std::shared_ptr<Tensor>> & tensors); //in: participating tensors identified by their names
 /** Builds a named tensor network from a template implemented by a custom tensor network builder. **/
 TensorNetwork(const std::string & name,                    //in: tensor network name
               std::shared_ptr<Tensor> output_tensor,       //in: output tensor of the tensor network
               NetworkBuilder & builder);                   //in: specific tensor network builder

 TensorNetwork(const TensorNetwork &) = default;
 TensorNetwork & operator=(const TensorNetwork &) = default;
 TensorNetwork(TensorNetwork &&) noexcept = default;
 TensorNetwork & operator=(TensorNetwork &&) noexcept = default;
 virtual ~TensorNetwork() = default;

 /** Prints **/
 void printIt() const;

 /** Returns TRUE if the tensor network is empty, FALSE otherwise. **/
 bool isEmpty() const;

 /** Returns TRUE if the tensor network is being built explicitly, FALSE otherwise. **/
 bool isExplicit() const;

 /** Returns TRUE if the tensor network is finalized, FALSE otherwise. **/
 bool isFinalized() const;

 /** Returns the number of input tensors in the tensor network.
     Note that the output tensor (tensor #0) is not counted here. **/
 unsigned int getNumTensors() const;

 /** Returns the maximal tensor id value used in the tensor network. **/
 unsigned int getMaxTensorId() const;

 /** Returns the name of the tensor network. **/
 const std::string & getName() const;

 /** Returns a given tensor of the tensor network without its connections (legs).
     If not found, returns nullptr. **/
 std::shared_ptr<Tensor> getTensor(unsigned int tensor_id);

 /** Get tensor connections. **/
 const std::vector<TensorLeg> * getTensorConnections(unsigned int tensor_id);

 /** Begin iterator **/
 inline Iterator begin() {return tensors_.begin();}
 /** End iterator **/
 inline Iterator end() {return tensors_.end();}
 /** Begin constant iterator **/
 inline ConstIterator cbegin() {return tensors_.cbegin();}
 /** End constant iterator **/
 inline ConstIterator cend() {return tensors_.cend();}

 /** Finalizes the explicit construction of the tensor network (construction with advance knowledge).
     The tensor network cannot be empty. **/
 bool finalize(bool check_validity = false);

 /** Appends a new tensor to the tensor network by matching the tensor modes
     with the modes of other tensors present or to be present in the tensor network.
     The fully specified output tensor with all its legs has had to be provided
     in advance in the TensorNetwork ctor. This way requires the advance knowledge
     of the entire tensor network. Once all tensors have been appended, one needs
     to call .finalize() to complete the construction of the tensor network. **/
 bool appendTensor(unsigned int tensor_id,                      //in: appended tensor id (unique within the tensor network)
                   std::shared_ptr<Tensor> tensor,              //in: appended tensor
                   const std::vector<TensorLeg> & connections); //in: tensor connections (fully specified)

 /** Appends a new tensor to the tensor network by matching the tensor modes
     with the modes of the output tensor of the tensor network. The unmatched modes
     of the newly appended tensor will be appended to the existing modes of the
     output tensor of the tensor network (at the end). The optional argument
     leg_dir allows specification of the leg direction for all tensor modes.
     If provided, the direction of the paired legs of the appended tensor
     must anti-match the direction of the corresponding legs of existing tensors. **/
 bool appendTensor(unsigned int tensor_id,                                                   //in: appended tensor id (unique within the tensor network)
                   std::shared_ptr<Tensor> tensor,                                           //in: appended tensor
                   const std::vector<std::pair<unsigned int, unsigned int>> & pairing,       //in: leg pairing: output tensor mode -> appended tensor mode
                   const std::vector<LegDirection> & leg_dir = std::vector<LegDirection>{}); //in: optional leg directions (for all tensor modes)

 /** Appends a new even-rank tensor to the tensor network by matching the first half
     of the tensor legs with network's output legs provided in "pairing". The second half
     of the tensor legs will then replace the matched output legs in the output tensor. **/
 bool appendTensorGate(unsigned int tensor_id,                     //in: appended tensor id (unique within the tensor network)
                       std::shared_ptr<Tensor> tensor,             //in: appended tensor gate (operator)
                       const std::vector<unsigned int> & pairing); //in: leg pairing: output tensor modes (half-rank)

 /** Appends a tensor network to the current (primary) tensor network by matching the modes
     of the output tensors of both tensor networks. The unmatched modes of the
     output tensor of the appended tensor network will be appended to the updated output
     tensor of the primary tensor network (at the end). The appended tensor network
     will cease to exist after being absorbed by the primary tensor network.
     If paired legs of either output tensor are directed, the directions must be respected.
     The tensors constituting the appended tensor network, except its output tensor,
     must have their unique ids be different from the ids of the tensors constituting
     the primary tensor network, otherwise the result is undefined and unrecoverable! **/
 bool appendTensorNetwork(TensorNetwork && network,                                            //in: appended tensor network
                          const std::vector<std::pair<unsigned int, unsigned int>> & pairing); //in: leg pairing: output tensor mode (primary) -> output tensor mode (appended)

 /** Appends an even-rank tensor network to the current (primary) tensor network by matching
     the first half of the output modes of the appended tensor network with selected
     modes of the current (primary) tensor network, simultaneously replacing the matched
     output modes of the current (primary) tensor network by the second half of the modes
     of the appended tensor network, going in order. Matching will respect leg directions.
     The replacing output modes of the appended tensor network mush have same directions
     as the replaced modes of the current (primary) tensor network. The appended tensor
     network will cease to exist after being absorbed by the primary tensor network.
     The tensors constituting the appended tensor network, except its output tensor,
     must have their unique ids be different from the ids of the tensors constituting
     the primary tensor network, otherwise the result is undefined and unrecoverable! **/
 bool appendTensorNetworkGate(TensorNetwork && network,                   //in: appended tensor network gate (operator)
                              const std::vector<unsigned int> & pairing); //in: leg pairing: output tensor modes of the primary network (half-rank)

 /** Reoders the modes of the output tensor of the tensor network:
     order[x] = y: yth mode of the output tensor becomes its xth mode. **/
 bool reoderOutputModes(const std::vector<unsigned int> & order); //in: new order of the output tensor modes (N2O)

 /** Deletes a tensor from a finalized tensor network (output tensor cannot be deleted).
     The released tensor legs will be joined at the end of the output tensor,
     unless a tensor leg was already connected to the output tensor, in which case
     it will be deleted completely, resulting in a reduced rank of the output tensor. **/
 bool deleteTensor(unsigned int tensor_id); //in: id of the tensor to be deleted

 /** Merges two tensors in a finalized tensor network by replacing them by their contracted product:
     result = left * right: All participating tensor ids must be distinct and not equal to 0.
     The uncontracted modes of the left tensor will precede in-order the uncontracted
     modes of the right tensor in the tensor-result. **/
 bool mergeTensors(unsigned int left_id,   //in: left tensor id (present in the tensor network)
                   unsigned int right_id,  //in: right tensor id (present in the tensor network)
                   unsigned int result_id, //in: result tensor id (absent in the tensor network, to be appended)
                   std::string * contr_pattern = nullptr); //inout: corresponding tensor contraction pattern (owned by the caller)

 /** Splits a given tensor in a finalized tensor network into two tensors by introducing new dimensions
     across the cutting boundary. The original tensor dimensions are then assigned to either left or
     right tensor. The new dimensions are then appended to both tensors at the end. The two tensors
     obtained via such splitting must get unique ids, one of them may be the original tensor_id. **/
 bool splitTensor(unsigned int tensor_id,                //in: id of the tensor to be split into two tensors
                  unsigned int left_tensor_id,           //in: id of the left tensor obtained via splitting
                  const std::string & left_tensor_name,  //in: name of the left tensor
                  unsigned int right_tensor_id,          //in: id of the right tensor obtained via splitting
                  const std::string & right_tensor_name, //in: name of the right tensor
                  const TensorShape & contracted_dims,   //in: dimension extents of the contracted (new) dimensions connecting two tensors after splitting
                  const std::vector<int> & right_dims);  //in: assignment of original tensor dimensions to new tensors (0: left, 1: right tensor)

 /** Returns the FMA flop count for a given contraction of two tensors identified by their ids
     in the tensor network. Optionally returns the arithmetic intensity of the tensor contraction as well.
     Additionally, it also allows rescaling of the tensor contraction cost with the adjustment
     by the arithmetic intensity (lower arithmetic intensity will increase the cost). **/
 double getContractionCost(unsigned int left_id,  //in: left tensor id (present in the tensor network)
                           unsigned int right_id, //in: right tensor id (present in the tensor network)
                           double * arithm_intensity = nullptr, //out: arithmetic intensity of the tensor contraction
                           bool adjust_cost = false); //in: whether or not to adjust the flops cost due to arithmetic intensity

 /** Returns the list of tensor operations required for evaluating the tensor network. **/
 std::list<std::shared_ptr<TensorOperation>> & getOperationList(const std::string & contr_seq_opt_name = "dummy");

protected:

 /** Returns a non-owning pointer to a given tensor of the tensor network
     together with its connections (legs). If not found, returns nullptr. **/
 TensorConn * getTensorConn(unsigned int tensor_id);

 /** Returns a vector of non-owning pointers to all tensors in the tensor network,
     except the output tensor. **/
 std::vector<TensorConn*> getTensorConnAll();

 /** Checks validity of connections of a given tensor. **/
 bool checkConnections(unsigned int tensor_id);
 /** Checks validity of connections in the enitre tensor network. **/
 bool checkConnections();

 /** Updates tensor network linking when a tensor has its connections modified:
     tensor_id is the id of the tensor whose leg numeration was updated. **/
 void updateConnections(unsigned int tensor_id); //in: id of the tensor whose connections were modified

 /** Invalidates cached tensor contraction sequence. **/
 void invalidateContractionSequence();

 /** Determines a pseudo-optimal tensor contraction sequence required for evaluating the tensor network.
     Returns an estimate of the total flop count required by the returned contraction sequence.
     The tensor network must contain at least two input tensors in order to generate a single contraction.
     No contraction sequence is generated for tensor networks consisting of a single input tensor. **/
 double determineContractionSequence(ContractionSeqOptimizer & contr_seq_optimizer);

private:

 int explicit_output_;                                  //whether or not the output tensor has been fully specified during construction
 int finalized_;                                        //finalization status of the tensor network
 std::string name_;                                     //tensor network name
 std::unordered_map<unsigned int, TensorConn> tensors_; //tensors connected to each other via legs (tensor connections)
                                                        //map: Non-negative tensor id --> Connected tensor
 double contraction_seq_flops_; //flop estimate for the determined tensor contraction sequence
 std::list<ContrTriple> contraction_seq_; //cached tensor contraction sequence
 std::list<std::shared_ptr<TensorOperation>> operations_; //cached tensor operations required for evaluating the tensor network
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_NETWORK_HPP_
