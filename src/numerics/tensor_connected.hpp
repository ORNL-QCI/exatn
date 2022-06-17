/** ExaTN::Numerics: Tensor connected to other tensors in a tensor network
REVISION: 2022/06/15

Copyright (C) 2018-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle)
Copyright (C) 2022-2022 NVIDA Corp. **/

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
#include "metadata.hpp"

#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <string>

#include "errors.hpp"

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
 void printIt(bool with_hash = false) const;
 void printItFile(std::ofstream & output_file,
                  bool with_hash = false) const;

 /** Returns the tensor name. **/
 const std::string & getName() const;

 /** Get the tensor shape. **/
 const TensorShape & getShape() const;

 /** Get the tensor signature. **/
 const TensorSignature & getSignature() const;

 /** Returns the total number of legs (tensor rank/order). **/
 unsigned int getNumLegs() const;
 unsigned int getRank() const;

 /** Returns the complex conjugation status of the tensor. **/
 bool isComplexConjugated() const;

 /** Returns a co-owned pointer to the tensor. **/
 std::shared_ptr<Tensor> getTensor() const;

 /** Returns the tensor id. **/
 unsigned int getTensorId() const;

 /** Resets the tensor id. **/
 void resetTensorId(unsigned int tensor_id);

 /** Returns a specific tensor leg. **/
 const TensorLeg & getTensorLeg(unsigned int leg_id) const;

 /** Returns all tensor legs. **/
 const std::vector<TensorLeg> & getTensorLegs() const;

 /** Returns the tensor dimension extents. **/
 const std::vector<DimExtent> & getDimExtents() const;

 /** Returns the dimension extent of a specific tensor leg. **/
 DimExtent getDimExtent(unsigned int dim_id) const;

 /** Get the space/subspace id for a specific tensor leg. **/
 std::pair<SpaceId,SubspaceId> getDimSpaceAttr(unsigned int dim_id) const;

 /** Resets an existing tensor leg (specific connection to another tensor). **/
 void resetLeg(unsigned int leg_id,   //in: leg id to reset
               TensorLeg tensor_leg); //in: new leg configuration

 /** Deletes an existing tensor leg, reducing the tensor rank by one. **/
 void deleteLeg(unsigned int leg_id);  //in: leg id to delete
 /** Deletes a set of existing tensor legs, reducing the tensor rank.
     The passed vector of leg ids will be sorted on return. **/
 void deleteLegs(std::vector<unsigned int> & leg_ids); //inout: vector of leg ids to delete

 /** Appends a new tensor leg as the last leg, increasing the tensor rank by one. **/
 void appendLeg(std::pair<SpaceId,SubspaceId> subspace, //in: subspace defining the new leg
                DimExtent dim_extent,                   //in: dimension extent of the new leg
                TensorLeg tensor_leg);                  //in: new leg configuration
 void appendLeg(DimExtent dim_extent,
                TensorLeg tensor_leg);

 /** Conjugates the connected tensor, which includes complex conjugation
     of the tensor itself as well as tensor leg direction reversal. **/
 void conjugate(); //changes the current conjugation status to the opposite
 void conjugate(bool conjug); //changes the conjugation status to the requested one

 /** Returns TRUE if the tensor is congruent to another tensor, that is,
     it has the same shape and signature. **/
 bool isCongruentTo(const TensorConn & another) const;

 /** Retrieves metadata attached to the connected tensor. **/
 const Metadata & getMetadata() const;

 /** Attaches metadata to the connected tensor. **/
 void attachMetadata(const Metadata & metadata);

 /** Retrieves a specific key-value pair from metadata attached to the connected tensor:
     Key is alphanumeric_ string, value is {integer/long, float/double, bool, string}.
     If key is found, returns true, otherwise false. **/
 template<typename ValueType>
 bool retrieveMetaValue(const std::string & key,
                        ValueType & value) const{
  return metadata_.retrieveValue(key,value);
 }

 /** Stores a given key-value pair in metadata attached to the connected tensor. **/
 template<typename ValueType>
 void appendMetaValue(const std::string & key,
                      const ValueType & value){
  return metadata_.appendKeyValue(key,value);
 }

 /** Clears metadata attached to the connected tensor. **/
 void clearMetadata(){
  return metadata_.clear();
 }

 /** Replaces the stored tensor with a new one (same shape and signature). **/
 void replaceStoredTensor(const std::string & name = ""); //in: tensor name (if empty, will be automatically generated)
 /** Replaces the stored tensor with a new one (permuted shape and signature). **/
 void replaceStoredTensor(const std::vector<unsigned int> & order, //in: new order of dimensions (N2O)
                          const std::string & name = ""); //in: tensor name (if empty, will be automatically generated)
 /** Replaces the stored tensor with a new one provided explicitly. **/
 void replaceStoredTensor(std::shared_ptr<Tensor> tensor);

 /** Queries whether the tensor has any isometries. **/
 bool hasIsometries() const;

 /** Retrieves the list of all registered isometries in the tensor. **/
 const std::list<std::vector<unsigned int>> & retrieveIsometries() const;

 /** Retrieves a specific group of isometric dimensions. **/
 std::vector<unsigned int> retrieveIsometry(unsigned int iso_group_id) const;

 /** Retrieves (in order) the tensor dimensions which
     do not belong to the specified isometric group. **/
 std::vector<unsigned int> retrieveIsometryComplement(unsigned int iso_group_id) const;

 /** Returns the ordered vector of non-isometric dimensions. **/
 std::vector<unsigned int> retrieveNonisometricDimensions() const;

 /** Unregisters all isometries in the tensor. **/
 void unregisterIsometries();

 /** Returns TRUE if the given tensor dimension belongs to a registered isometry group. **/
 bool withIsometricDimension(unsigned int dim_id,                                           //in: tensor dimension id
                             const std::vector<unsigned int> ** iso_group = nullptr) const; //out: pointer to the registered isometric group

 /** Returns whether this connected tensor is optimizable or not (whether or not this
     connected tensor should be optimized during the tensor network functional optimization).**/
 bool isOptimizable() const;

 /** Resets the optimizability attribute (whether or not this connected tensor
     should be optimized during the tensor network functional optimization).
     Note that this attribute specifically applies to the tensor in its current
     connected position within the tensor network, not to the tensor per se.
     The output tensor of the tensor network (id = 0) cannot be optimizable. **/
 void resetOptimizability(bool optimizable);

 /** Returns the tensor element type. **/
 TensorElementType getElementType() const;

private:

 std::shared_ptr<Tensor> tensor_; //co-owned pointer to the tensor
 unsigned int id_;                //tensor id in the tensor network
 std::vector<TensorLeg> legs_;    //tensor legs: Connections to other tensors
 Metadata metadata_;              //tensor metadata
 bool conjugated_;                //complex conjugation flag
 bool optimizable_;               //whether or not the tensor is subject to optimization as part of the optimized tensor network
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_CONNECTED_HPP_
