/** ExaTN::Numerics: Tensor network
REVISION: 2019/07/07

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
               (1) Matching the tensor modes with the modes of the input tensors
                   already present in the tensor network.
               (2) Matching the tensor modes with the modes of the output tensor
                   of the tensor network.
               In both cases, the unmatched modes of the newly appended tensor
               will be appended to the output tensor of the tensor network,
               succeeding the existing modes of the output tensor.
     Option 2: A tensor network can be appended to another tensor network by
               matching the modes of the output tensors of both tensor networks.
               The unmatched modes of the output tensor of the appended tensor
               network will be appended to the output tensor of the primary
               tensor network (at the end). The appended tensor network will
               cease to exist after being absorbed by the primary tensor network.
 (e) The modes of the output tensor of a tensor network can be examined and reordered.
**/

#ifndef EXATN_NUMERICS_TENSOR_NETWORK_HPP_
#define EXATN_NUMERICS_TENSOR_NETWORK_HPP_

#include "tensor_basic.hpp"
#include "tensor_connected.hpp"
#include "tensor_op_factory.hpp"

#include <unordered_map>
#include <vector>
#include <string>
#include <memory>

namespace exatn{

namespace numerics{

class TensorNetwork{
public:

 static constexpr unsigned int NUM_WALKERS = 1024; //default number of walkers for tensor contraction sequence optimization

 using ContractionSequence = std::vector<std::pair<unsigned int, unsigned int>>; //pairs of contracted tensor id's

 /** Creates an unnamed empty tensor network with a single scalar output tensor named "_SMOKY_TENSOR_" **/
 TensorNetwork();
 /** Creates a named empty tensor network with a single scalar output tensor named with the same name. **/
 TensorNetwork(const std::string & name);

 TensorNetwork(const TensorNetwork &) = default;
 TensorNetwork & operator=(const TensorNetwork &) = default;
 TensorNetwork(TensorNetwork &&) noexcept = default;
 TensorNetwork & operator=(TensorNetwork &&) noexcept = default;
 virtual ~TensorNetwork() = default;

 /** Prints **/
 void printIt() const;

 /** Returns TRUE if the tensor network is empty, FALSE otherwise. **/
 bool isEmpty() const;

 /** Returns the number of input tensors in the tensor network.
     Note that the output tensor (tensor #0) is not counted here. **/
 unsigned int getNumTensors() const;

 /** Returns the name of the tensor network. **/
 const std::string & getName() const;

 /** Returns a non-owning pointer to a given tensor of the tensor network
     together with its connections (legs). If not found, returns nullptr. **/
 const TensorConn * getTensorConn(unsigned int tensor_id) const;

 /** Returns a given tensor of the tensor network without its connections (legs).
     If not found, returns nullptr. **/
 std::shared_ptr<Tensor> getTensor(unsigned int tensor_id);

 /** Appends a new tensor to the tensor network by matching the tensor modes
     with the modes of the input tensors already present in the tensor network.
     The unmatched modes of the newly appended tensor will be appended to the
     existing modes of the output tensor of the tensor network (at the end). **/
 bool appendTensor(unsigned int tensor_id,                      //in: tensor id (unique within the tensor network)
                   std::shared_ptr<Tensor> tensor,              //in: appended tensor
                   const std::vector<TensorLeg> & connections); //in: tensor connections

 /** Appends a new tensor to the tensor network by matching the tensor modes
     with the modes of the output tensor of the tensor network. The unmatched modes
     of the newly appended tensor will be appended to the existing modes of the
     output tensor of the tensor network (at the end). **/
 bool appendTensor(unsigned int tensor_id,                                              //in: tensor id (unique within the tensor network)
                   std::shared_ptr<Tensor> tensor,                                      //in: appended tensor
                   const std::vector<std::pair<unsigned int, unsigned int>> & pairing); //in: leg pairing: output tensor mode -> appended tensor mode

 /** Appends a tensor network to the current tensor network by matching the modes
     of the output tensors of both tensor networks. The unmatched modes of the
     output tensor of the appended tensor network will be appended to the output
     tensor of the primary tensor network (at the end). The appended tensor network
     will cease to exist after being absorbed by the primary tensor network. **/
 bool appendTensorNetwork(TensorNetwork && network,                                            //in: appended tensor network
                          const std::vector<std::pair<unsigned int, unsigned int>> & pairing); //in: leg pairing: output tensor mode (primary) -> output tensor mode (appended)

 /** Reoders the modes of the output tensor of the tensor network. **/
 bool reoderOutputModes(const std::vector<unsigned int> & order); //in: new order of the output tensor modes (N2O)

private:

 std::string name_;                                     //tensor network name
 std::unordered_map<unsigned int, TensorConn> tensors_; //tensors connected to each other via legs (tensor connections)
                                                        //map: Non-negative tensor id --> Connected tensor
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_NETWORK_HPP_
