/** ExaTN::Numerics: Tensor network
REVISION: 2020/06/01

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

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

#include <functional>
#include <unordered_map>
#include <map>
#include <vector>
#include <list>
#include <tuple>
#include <string>
#include <memory>

namespace exatn{

namespace numerics{

//Index (dimension) split information: Vector of segments the full dimension is split into:
using IndexSplit = std::vector<std::pair<SubspaceId, DimExtent>>; //Segment = [subspace_base, segment_extent]


//Tests whether a given tensor is an intermediate tensor of a tensor network:
bool tensorIsIntermediate(const Tensor & tensor,            //in: tensor
                          bool * network_output = nullptr); //out: TRUE if the tensor is the output tensor of the tensor network

//Free function analogue of TensorNetwork::getContractionCost:
double getTensorContractionCost(const TensorConn & left_tensor,
                                const TensorConn & right_tensor,
                                double * diff_volume = nullptr,
                                double * arithm_intensity = nullptr,
                                bool adjust_cost = false);


class TensorNetwork{

 friend class TensorExpansion;

public:

 using Iterator = typename std::unordered_map<unsigned int, TensorConn>::iterator; //iterator
 using ConstIterator = typename std::unordered_map<unsigned int, TensorConn>::const_iterator; //constant iterator

 /** Creates an unnamed empty tensor network with a single scalar output tensor named "_smoky" **/
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
 /** Clones a tensor network with an optional replacement of the output tensor. **/
 TensorNetwork(const TensorNetwork & another,               //in: another tensor network
               bool replace_output,                         //in: whether or not to replace the output tensor
               const std::string & new_output_name = "");   //in: new name of the output tensor (if empty, will be generated)

 TensorNetwork(const TensorNetwork &) = default;
 TensorNetwork & operator=(const TensorNetwork &) = default;
 TensorNetwork(TensorNetwork &&) noexcept = default;
 TensorNetwork & operator=(TensorNetwork &&) noexcept = default;
 virtual ~TensorNetwork() = default;

 /** Prints **/
 void printIt(bool with_tensor_hash = false) const;

 /** Returns TRUE if the tensor network is empty, FALSE otherwise. **/
 bool isEmpty() const;

 /** Returns TRUE if the tensor network is being built explicitly, FALSE otherwise. **/
 bool isExplicit() const;

 /** Returns TRUE if the tensor network is finalized, FALSE otherwise. **/
 bool isFinalized() const;

 /** Returns TRUE if the tensor network is valid, FALSE otherwise. **/
 bool isValid();

 /** Returns the rank of the tensor network (rank of its output tensor). **/
 unsigned int getRank() const;

 /** Returns the number of input tensors in the tensor network.
     Note that the output tensor (tensor #0) is not counted here. **/
 unsigned int getNumTensors() const;

 /** Returns the maximal tensor id value used in the tensor network. **/
 unsigned int getMaxTensorId();

 /** Returns the name of the tensor network. **/
 const std::string & getName() const;

 /** Renames the tensor network. The output tensor is reset to a new one as well. **/
 void rename(const std::string & name);

 /** Returns a given tensor of the tensor network without its connections (legs).
     If not found, returns nullptr. **/
 std::shared_ptr<Tensor> getTensor(unsigned int tensor_id,
                                   bool * conjugated = nullptr) const;

 /** Returns tensor connections. **/
 const std::vector<TensorLeg> * getTensorConnections(unsigned int tensor_id) const;

 /** Returns a list of the tensors adjacent to a given tensor by their Ids. **/
 std::list<unsigned int> getAdjacentTensors(unsigned int tensor_id) const;

 /** Begin iterator **/
 inline Iterator begin() {return tensors_.begin();}
 /** End iterator **/
 inline Iterator end() {return tensors_.end();}
 /** Begin constant iterator **/
 inline ConstIterator cbegin() const {return tensors_.cbegin();}
 /** End constant iterator **/
 inline ConstIterator cend() const {return tensors_.cend();}

 /** Finalizes the explicit construction of the tensor network (construction with advance knowledge).
     The tensor network cannot be empty. **/
 bool finalize(bool check_validity = false);

 /** Appends a new tensor to the tensor network by matching the tensor modes
     with the modes of other tensors present or to be present in the tensor network.
     The fully specified output tensor with all its legs has had to be provided
     in advance in the TensorNetwork ctor. This way requires the advance knowledge
     of the entire tensor network. Once all tensors have been appended, one needs
     to call .finalize() to complete the construction of the tensor network. **/
 bool placeTensor(unsigned int tensor_id,                     //in: appended tensor id (unique within the tensor network)
                  std::shared_ptr<Tensor> tensor,             //in: appended tensor
                  const std::vector<TensorLeg> & connections, //in: tensor connections (fully specified)
                  bool conjugated = false,                    //in: complex conjugation flag for the appended tensor
                  bool leg_matching_check = true);            //in: tensor leg matching check

 /** Appends a new tensor to the tensor network by matching the tensor modes
     with the modes of the output tensor of the tensor network. The unmatched modes
     of the newly appended tensor will be appended to the existing modes of the
     output tensor of the tensor network (at the end). The optional argument
     leg_dir allows specification of the leg direction for all tensor modes.
     If provided, the direction of the paired legs of the appended tensor
     must anti-match the direction of the corresponding legs of existing tensors. **/
 bool appendTensor(unsigned int tensor_id,                                                  //in: appended tensor id (unique within the tensor network)
                   std::shared_ptr<Tensor> tensor,                                          //in: appended tensor
                   const std::vector<std::pair<unsigned int, unsigned int>> & pairing,      //in: leg pairing: output tensor mode -> appended tensor mode
                   const std::vector<LegDirection> & leg_dir = std::vector<LegDirection>{}, //in: optional leg directions (for all tensor modes)
                   bool conjugated = false);                                                //in: complex conjugation flag for the appended tensor

 /** Appends a new even-rank tensor to the tensor network by matching the first half
     of the tensor legs with network's output legs provided in "pairing". The second half
     of the tensor legs will then replace the matched output legs in the output tensor. **/
 bool appendTensorGate(unsigned int tensor_id,                    //in: appended tensor id (unique within the tensor network)
                       std::shared_ptr<Tensor> tensor,            //in: appended tensor gate (operator)
                       const std::vector<unsigned int> & pairing, //in: leg pairing: output tensor modes (half-rank)
                       bool conjugated = false);                  //in: complex conjugation flag for the appended tensor gate

 /** Appends a tensor network to the current (primary) tensor network by matching the modes
     of the output tensors of both tensor networks. The unmatched modes of the output tensor
     of the appended tensor network will be appended to the updated output tensor of the
     primary tensor network (at the end). The appended tensor network will cease to exist
     after being absorbed by the primary tensor network. If paired legs of either output
     tensor are directed, the directions must be respected. The tensors constituting the
     appended tensor network, except its output tensor, will obtain their unique ids to be
     different from the ids of the tensors constituting the primary tensor network. **/
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
     will obtain their unique ids to be different from the ids of the tensors constituting
     the primary tensor network. **/
 bool appendTensorNetworkGate(TensorNetwork && network,                   //in: appended tensor network gate (operator)
                              const std::vector<unsigned int> & pairing); //in: leg pairing: output tensor modes of the primary network (half-rank)

 /** Reorders the modes of the output tensor of the tensor network:
     order[x] = y: yth mode of the output tensor becomes its xth mode. **/
 bool reorderOutputModes(const std::vector<unsigned int> & order); //in: new order of the output tensor modes (N2O)

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

 /** Substitutes a tensor in the tensor network with another congruent tensor. **/
 bool substituteTensor(unsigned int tensor_id,          //in: id of the tensor to be substituted
                       std::shared_ptr<Tensor> tensor); //in: substituting tensor
 bool substituteTensor(const std::string & name,        //in: name of the tensor to be substituted
                       std::shared_ptr<Tensor> tensor); ////in: substituting tensor

 /** Returns the list of id's the given tensor enters the tensor network with. **/
 std::vector<unsigned int> getTensorIdsInNetwork(const std::string & name,       //in: tensor name
                                                 bool conjugated = false) const; //in: whether or not look for conjugated tensors with the given name

 /** Conjugates the tensor network, which includes complex conjugation of
     all tensors as well as reversal of the direction of all tensor legs. **/
 bool conjugate();

 /** Collapses all isometric tensor pairs, thus simplifying the tensor network.
     Returns TRUE if at least one isometric tensor pair has been collapsed.
     An isometric tensor pair is a pair of a tensor and its conjugate which
     are solely contracted over exactly one of their isometric dimension groups
     while all other contracted dimensions of both tensors must involve other tensors.
     Note that an isometric collapse may introduce trace legs in the remaining tensors
     of the tensor network in case both tensors from the isometric tensor pair were contracted
     with the same tensor via the same subset of tensor dimensions. In this case, make sure
     that the tensor processing runtime of your choice supports tensor tracing, or, in case
     of the output tensor it should be able to handle spectators (orphaned tensor legs). **/
 bool collapseIsometries();

 /** Decomposes all tensors in the tensor network to restrict the highest tensor order to 3. **/
 bool decomposeTensors();

 /** Partitions the tensor network into multiple parts by minimizing the weighted edge cut.
     The returned vector <parts> is:
      parts[i] = pair{Partition weight, Ordered list of vertices forming partition i}.
     Note that the true logarithmic_2 edge cut = (edge_cut - num_cross_edges). **/
 bool partition(std::size_t num_parts, //in: desired number of parts
                double imbalance,      //in: tolerated partition weight imbalance
                std::vector<std::pair<std::size_t,std::vector<std::size_t>>> & parts, //out: partitions
                std::size_t * edge_cut = nullptr, //out: achieved edge cut value
                std::size_t * num_cross_edges = nullptr) const; //out: total number of cross edges

 /** Traverses the tensor network and marks certain tensors as optimizable
     based on the user-provided predicate function. If marked optimizable,
     these specific tensors (in their specific positions within the tensor network)
     will become subject to optimization when optimizing the tensor network. **/
 void markOptimizableTensors(std::function<bool (const Tensor &)> predicate);

 /** Returns the FMA flop count for a given contraction of two tensors identified by their ids
     in the tensor network. Optionally returns the volume difference. Optionally also returns
     the arithmetic intensity of the tensor contraction. Additionally, it also allows rescaling
     of the tensor contraction cost with the adjustment by the arithmetic intensity (lower
     arithmetic intensity will effectively increase the flop cost). **/
 double getContractionCost(unsigned int left_id,  //in: left tensor id (present in the tensor network)
                           unsigned int right_id, //in: right tensor id (present in the tensor network)
                           double * diff_volume = nullptr, //out: vol(result) - vol(left) - vol(right)
                           double * arithm_intensity = nullptr, //out: arithmetic intensity of the tensor contraction
                           bool adjust_cost = false); //in: whether or not to adjust the flops cost due to arithmetic intensity

 /** Imports and caches an externally provided tensor contraction sequence. **/
 void importContractionSequence(const std::list<ContrTriple> & contr_sequence);

 /** Returns the currently stored tensor contraction sequence, if any. **/
 const std::list<ContrTriple> & exportContractionSequence() const;

 /** Returns the list of tensor operations required for evaluating the tensor network.
     Parameter universal_indices set to TRUE will activate the universal index numeration
     such that a specific index appearing in different tensor operations will always
     designate the same edge in the tensor network, and all tensors will carry real names. **/
 std::list<std::shared_ptr<TensorOperation>> & getOperationList(const std::string & contr_seq_opt_name = "metis",
                                                                bool universal_indices = false);

 /** Splits some indices of the tensor network into smaller segments in order
     to make sure all intermediates from the operation list will fit within
     the given memory limit. The generated information will be available to
     the processing backend when the tensor network is submitted for evaluation. **/
 void splitIndices(std::size_t max_intermediate_volume); //in: intermediate volume limit

 /** Returns the total number of splitted indices. **/
 unsigned int getNumSplitIndices() const;

 /** Returns the splitting information for a chosen global index. **/
 const std::pair<std::string,IndexSplit> &
 getSplitIndexInfo(unsigned int global_index_id) const;

 /** Returns the splitting information for a given tensor from the operation list. **/
 const std::vector<std::pair<unsigned int, unsigned int>> *
 getSplitTensorInfo(const std::pair<TensorHashType,TensorHashType> & key) const;

 /** Prints information on index splitting within the tensor operation list. **/
 void printSplitIndexInfo(bool with_affected_tensors = false) const;

 /** Returns the FMA flop count estimate required for evaluating the tensor network,
     if available (if getOperationList has already been invoked). **/
 double getFMAFlops() const;

 /** Returns the maximal cumulative volume of intermediate tensors present at a time. **/
 double getMaxIntermediatePresenceVolume() const;

 /** Returns the volume of the largest intermediate tensor required for evaluating
     the tensor network, if available (if getOperationList has already been invoked). **/
 double getMaxIntermediateVolume(unsigned int * intermediate_rank = nullptr) const;

 /** Returns the entire tensor network printed in a symbolic form.
     The tensor network must already have its operation list generated. **/
 bool printTensorNetwork(std::string & network);

 /** Returns a non-owning pointer to a given tensor of the tensor network
     together with its connections (legs). If not found, returns nullptr. **/
 TensorConn * getTensorConn(unsigned int tensor_id);

protected:

 /** Emplaces a connected tensor into the tensor network. **/
 inline bool emplaceTensorConn(unsigned int tensor_id,
                               const TensorConn & tensor_conn);
 inline bool emplaceTensorConn(bool dynamic_id_enabled,
                               unsigned int tensor_id,
                               const TensorConn & tensor_conn); //tensor_id may change if dynamic_id_enabled

 /** Emplaces a connected tensor into the tensor network. **/
 template <typename... Args>
 inline bool emplaceTensorConnDirect(bool dynamic_id_enabled,
                                     unsigned int tensor_id,
                                     Args&&... args); //arguments for TensorConn ctor
 template <typename... Args>
 inline bool emplaceTensorConnDirect(bool dynamic_name_enabled,
                                     bool dynamic_id_enabled,
                                     unsigned int tensor_id,
                                     Args&&... args); //arguments for TensorConn ctor

 /** Erases a connected tensor from the tensor network. **/
 inline bool eraseTensorConn(unsigned int tensor_id);

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

 /** Calls updateConnections() method for all input tensors.
     This is used for updating the output tensor legs. **/
 void updateConnectionsFromInputTensors();

 /** Invalidate the cached max tensor id. **/
 void invalidateMaxTensorId();

 /** Invalidates cached tensor contraction sequence. **/
 void invalidateContractionSequence();

 /** Determines a pseudo-optimal tensor contraction sequence required for evaluating the tensor network.
     Returns an estimate of the total flop count required by the returned contraction sequence.
     The tensor network must contain at least two input tensors in order to generate a single contraction.
     No contraction sequence is generated for tensor networks consisting of a single input tensor. **/
 double determineContractionSequence(ContractionSeqOptimizer & contr_seq_optimizer);

 /** Establishes a universal index numeration in the already generated tensor operation list
     such that a specific index occuring in different tensor operations will always refer
     to the same edge in the tensor network. It will also assure the use of real tensor names.
     If the tensor operation list is empty, does nothing. **/
 void establishUniversalIndexNumeration();

private:

 /** Resets the output tensor in a finalized tensor network to a new
     one with the same signature and shape but a different name. **/
 void resetOutputTensor(const std::string & name = ""); //in: new name of the output tensor (if empty, will be generated automatically)

 /** Resets the output tensor in a finalized tensor network to a new
     one with a permuted signature and shape, and a different name. **/
 void resetOutputTensor(const std::vector<unsigned int> & order, //in: new order of dimensions (N2O)
                        const std::string & name = ""); //in: new name of the output tensor (if empty, will be generated automatically)

 /** Updates the max tensor id used in the tensor network when a tensor
     is either appended to or removed from the tensor network.  **/
 void updateMaxTensorIdOnAppend(unsigned int tensor_id);
 void updateMaxTensorIdOnRemove(unsigned int tensor_id);

 /** Data members: Core: **/
 int explicit_output_;                                  //whether or not the output tensor has been fully specified during construction
 int finalized_;                                        //finalization status of the tensor network
 std::string name_;                                     //tensor network name
 std::unordered_map<unsigned int, TensorConn> tensors_; //tensors connected to each other via legs (tensor connections)
                                                        //map: Non-negative tensor id --> Connected tensor
 /** Data members: Tensor id management: **/
 unsigned int max_tensor_id_;   //cached max tensor id used so far (0:undefined)

 /** Data members: Contraction sequence: **/
 double contraction_seq_flops_; //flop estimate for the determined tensor contraction sequence
 double max_intermediate_presence_volume_; //max cumulative volume of intermediates present at a time
 double max_intermediate_volume_; //volume of the largest intermediate tensor
 unsigned int max_intermediate_rank_; //rank of the largest intermediate tensor
 std::list<ContrTriple> contraction_seq_; //cached tensor contraction sequence
 std::list<std::shared_ptr<TensorOperation>> operations_; //cached tensor operations required for evaluating the tensor network
 std::vector<std::pair<std::string, //universal (unique) label of the index that was split
                       IndexSplit>  //information on the segments the index is split into
            > split_indices_;       //internal tensor network indices which were split
 std::map<std::pair<TensorHashType,  //tensor operation identifier (hash), only for input tensors (0 for intermediates)
                    TensorHashType>, //tensor operand identifier (tensor hash for intermediates, tensor operand position for input tensors)
          std::vector<std::pair<unsigned int,  //global id of the split index (in split_indices_): [0..max]
                                unsigned int>> //position of the split index in the tensor operand: [0..max]
         > split_tensors_; //information on tensors with split dimensions
 bool universal_indexing_; //universal indexing flag
};


//DEFINITIONS:
inline bool TensorNetwork::emplaceTensorConn(unsigned int tensor_id,
                                             const TensorConn & tensor_conn)
{
 auto res = tensors_.emplace(tensor_id,tensor_conn);
 if(res.second){
  res.first->second.resetTensorId(tensor_id);
  updateMaxTensorIdOnAppend(tensor_id);
 }
 return res.second;
}


inline bool TensorNetwork::emplaceTensorConn(bool dynamic_id_enabled,
                                             unsigned int tensor_id,
                                             const TensorConn & tensor_conn)
{
 auto res = tensors_.emplace(tensor_id,tensor_conn);
 if(!(res.second) && dynamic_id_enabled){
  tensor_id = getMaxTensorId() + 1;
  assert(tensor_id != 0); //unsigned int overflow
  res = tensors_.emplace(tensor_id,tensor_conn);
 }
 if(res.second){
  res.first->second.resetTensorId(tensor_id);
  updateMaxTensorIdOnAppend(tensor_id);
 }
 return res.second;
}


template <typename... Args>
inline bool TensorNetwork::emplaceTensorConnDirect(bool dynamic_id_enabled,
                                                   unsigned int tensor_id,
                                                   Args&&... args)
{
 auto res = tensors_.emplace(tensor_id,TensorConn(std::forward<Args>(args)...));
 if(!(res.second) && dynamic_id_enabled){
  tensor_id = getMaxTensorId() + 1;
  assert(tensor_id != 0); //unsigned int overflow
  res = tensors_.emplace(tensor_id,TensorConn(std::forward<Args>(args)...));
 }
 if(res.second){
  res.first->second.resetTensorId(tensor_id);
  updateMaxTensorIdOnAppend(tensor_id);
 }
 return res.second;
}


template <typename... Args>
inline bool TensorNetwork::emplaceTensorConnDirect(bool dynamic_name_enabled,
                                                   bool dynamic_id_enabled,
                                                   unsigned int tensor_id,
                                                   Args&&... args)
{
 auto res = tensors_.emplace(tensor_id,TensorConn(std::forward<Args>(args)...));
 if(!(res.second) && dynamic_id_enabled){
  tensor_id = getMaxTensorId() + 1;
  assert(tensor_id != 0); //unsigned int overflow
  res = tensors_.emplace(tensor_id,TensorConn(std::forward<Args>(args)...));
 }
 if(res.second){
  res.first->second.resetTensorId(tensor_id);
  updateMaxTensorIdOnAppend(tensor_id);
  if(dynamic_name_enabled){
   auto & stored_tensor = *(res.first->second.getTensor());
   stored_tensor.rename(generateTensorName(stored_tensor,"x")); //intermediate tensor prefix "x": _xHASH
  }
 }
 return res.second;
}


inline bool TensorNetwork::eraseTensorConn(unsigned int tensor_id)
{
 auto num_deleted = tensors_.erase(tensor_id);
 if(num_deleted == 1) updateMaxTensorIdOnRemove(tensor_id);
 return (num_deleted == 1);
}

} //namespace numerics

/** Creates a new tensor network as a shared pointer. **/
template<typename... Args>
inline std::shared_ptr<numerics::TensorNetwork> makeSharedTensorNetwork(Args&&... args)
{
 return std::make_shared<numerics::TensorNetwork>(std::forward<Args>(args)...);
}

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_NETWORK_HPP_
