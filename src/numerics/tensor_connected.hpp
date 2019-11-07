/** ExaTN::Numerics: Tensor connected to other tensors in a tensor network
REVISION: 2019/11/07

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) A tensor inside a tensor network is generally connected
     to other tensors in that network via so-called tensor legs;
     each tensor leg is associated with a specific tensor dimension.
 (b) Each tensor leg specifies a connection of a given tensor dimension
     to some dimension (or dimensions) in another tensor (or tensors) in
     the same tensor network. Thus, tensor legs can be binary, ternary, etc.,
     based on whether the tensor network is a graph or a hyper-graph.
 (c) The abstraction of a connected tensor is introduced for a quick
     inspection of the neighborhood of a chosen tensor inside the tensor network.
**/

#ifndef EXATN_NUMERICS_TENSOR_CONNECTED_HPP_
#define EXATN_NUMERICS_TENSOR_CONNECTED_HPP_

#include "tensor_basic.hpp"
#include "tensor_leg.hpp"
#include "tensor.hpp"

#include <memory>
#include <vector>

namespace exatn{

namespace numerics{

class TensorConn{
public:

 /** Constructs a connected tensor inside a tensor network. **/
 TensorConn(std::shared_ptr<Tensor> tensor,      //in: co-owned pointer to the tensor
            unsigned int id,                     //in: tensor id in the tensor network
            const std::vector<TensorLeg> & legs, //in: tensor legs: Connections to other tensors in the tensor network
            bool conjugated = false);            //in: whether or not the tensor enters a tensor network as complex conjugated

 TensorConn(const TensorConn &) = default;
 TensorConn & operator=(const TensorConn &) = default;
 TensorConn(TensorConn &&) noexcept = default;
 TensorConn & operator=(TensorConn &&) noexcept = default;
 virtual ~TensorConn() = default;

 /** Prints. **/
 void printIt() const;

 /** Returns the total number of legs (tensor rank/order). **/
 unsigned int getNumLegs() const;

 /** Returns the complex conjugation status of the tensor. **/
 bool isComplexConjugated() const;

 /** Returns a co-owned pointer to the tensor. **/
 std::shared_ptr<Tensor> getTensor();

 /** Returns the tensor id. **/
 unsigned int getTensorId() const;

 /** Resets the tensor id. **/
 void resetTensorId(unsigned int tensor_id);

 /** Returns a specific tensor leg. **/
 const TensorLeg & getTensorLeg(unsigned int leg_id) const;

 /** Returns all tensor legs. **/
 const std::vector<TensorLeg> & getTensorLegs() const;

 /** Returns the dimension extent of a specific tensor leg. **/
 DimExtent getDimExtent(unsigned int dim_id) const;

 /** Get the space/subspace id for a specific tensor leg. **/
 std::pair<SpaceId,SubspaceId> getDimSpaceAttr(unsigned int dim_id) const;

 /** Resets an existing tensor leg (specific connection to another tensor). **/
 void resetLeg(unsigned int leg_id,   //in: leg id to reset
               TensorLeg tensor_leg); //in: new leg configuration

 /** Deletes an existing tensor leg, reducing the tensor rank by one. **/
 void deleteLeg(unsigned int leg_id);  //in: leg id to delete
 /** Deletes a set of existing tensor legs, reducing the tensor rank. **/
 void deleteLegs(std::vector<unsigned int> & leg_ids); //inout: vector of leg ids to delete

 /** Appends a new tensor leg as the last leg, increasing the tensor rank by one. **/
 void appendLeg(std::pair<SpaceId,SubspaceId> subspace, //in: subspace defining the new leg
                DimExtent dim_extent,                   //in: dimension extent of the new leg
                TensorLeg tensor_leg);                  //in: new leg configuration
 void appendLeg(DimExtent dim_extent,
                TensorLeg tensor_leg);

 /** Conjugates the connected tensor, which includes complex conjugation
     of the tensor itself as well as tensor leg direction reversal. **/
 void conjugate();

private:

 std::shared_ptr<Tensor> tensor_; //co-owned pointer to the tensor
 unsigned int id_;                //tensor id in the tensor network
 std::vector<TensorLeg> legs_;    //tensor legs: Connections to other tensors
 bool conjugated_;                //complex conjugation flag
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_CONNECTED_HPP_
