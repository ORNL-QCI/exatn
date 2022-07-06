/** ExaTN::Numerics: Abstract Tensor
REVISION: 2022/06/07

Copyright (C) 2018-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle)
Copyright (C) 2022-2022 NVIDIA Corporation

SPDX-License-Identifier: BSD-3-Clause **/

/** NOTES:
 Tensor specification requires:
 (a) Symbolic tensor name;
 (b) Tensor rank (number of tensor dimensions) and
     tensor shape (extents of all tensor dimensions);
 (c) Optional tensor signature (space/subspace identifier for all tensor dimensions).
 (d) Optional tensor element type (exatn::TensorElementType).
 (e) Optional isometries: An isometry is a group of tensor dimensions a contraction over
     which with the conjugated tensor results in a delta function over the remaining
     tensor dimensions, spit between the original and conjugated tensors.
     A tensor is isometric if it has at least one isometry group of dimensions.
     A tensor is unitary if its dimensions can be partioned into two non-overlapping
     groups such that both groups form an isometry (in this case the volumes of both
     dimension groups will necessarily be the same).

 Tensor signature identifies a full tensor or its slice. Tensor signature
 requires providing a pair<SpaceId,SubspaceId> for each tensor dimension.
 It has two alternative specifications:
 (a) SpaceId = SOME_SPACE: In this case, SubspaceId is the lower bound
     of the specified tensor slice (0 is the min lower bound). The upper
     bound is computed by adding the dimension extent to the lower bound - 1.
     The defining vector space (SOME_SPACE) is an abstract anonymous vector space.
 (b) SpaceId != SOME_SPACE: In this case, SpaceId refers to a registered
     vector space and the SubspaceId refers to a registered subspace of
     this vector space. The subspaces will carry lower/upper bounds of
     the specified tensor slice. SubspaceId = 0 refers to the full space,
     which is automatically registered when the space is registered.
     Although tensor dimension extents cannot exceed the dimensions
     of the corresponding registered subspaces from the tensor signature,
     they in general can be smaller than the latter (low-rank representation).
**/

#ifndef EXATN_NUMERICS_TENSOR_HPP_
#define EXATN_NUMERICS_TENSOR_HPP_

#include "tensor_basic.hpp"
#include "packable.hpp"
#include "tensor_shape.hpp"
#include "tensor_signature.hpp"
#include "tensor_leg.hpp"

#include <iostream>
#include <fstream>
#include <type_traits>
#include <string>
#include <initializer_list>
#include <vector>
#include <list>
#include <memory>

#include "errors.hpp"

namespace exatn{

namespace numerics{

using TensorHashType = std::size_t;

class Tensor: public Packable {
public:

 /** Create a tensor by providing its name, shape and signature. **/
 Tensor(const std::string & name,           //tensor name
        const TensorShape & shape,          //tensor shape
        const TensorSignature & signature); //tensor signature
 /** Create a tensor by providing its shape and signature,
     the name will be generated automatically based on the tensor hash. **/
 Tensor(const TensorShape & shape,          //tensor shape
        const TensorSignature & signature); //tensor signature
 /** Create a tensor by providing its name and shape.
     The signature defaults to SOME_SPACE spaces and lbound=0 subspaces. **/
 Tensor(const std::string & name,           //tensor name
        const TensorShape & shape);         //tensor shape
 /** Create a tensor by providing its shape, the name will be generated
     automatically based on the tensor hash. **/
 Tensor(const TensorShape & shape);         //tensor shape
 /** Create a tensor by providing its name, shape and signature from scratch. **/
 template<typename T>
 Tensor(const std::string & name,                                        //tensor name
        std::initializer_list<T> extents,                                //tensor dimension extents
        std::initializer_list<std::pair<SpaceId,SubspaceId>> subspaces); //tensor dimension defining subspaces
 template<typename T>
 Tensor(const std::string & name,                                      //tensor name
        const std::vector<T> & extents,                                //tensor dimension extents
        const std::vector<std::pair<SpaceId,SubspaceId>> & subspaces); //tensor dimension defining subspaces
 /** Create a tensor by providing its name and shape from scratch.
     The signature defaults to SOME_SPACE spaces and lbound=0 subspaces. **/
 template<typename T>
 Tensor(const std::string & name,          //tensor name
        std::initializer_list<T> extents); //tensor dimension extents
 template<typename T>
 Tensor(const std::string & name,        //tensor name
        const std::vector<T> & extents); //tensor dimension extents
 /** Create a rank-0 tensor (scalar). **/
 Tensor(const std::string & name);       //tensor name
 /** Create a tensor by contracting two other tensors.
     The vector of tensor legs specifies the tensor contraction pattern:
      contraction[] describes dimensions of both input tensors,
      first left tensor dimensions, then right tensor dimensions:
      contraction.size() = left_rank + right_rank;
      Output tensor id = 0;
      Left input tensor id = 1;
      Right input tensor id = 2. **/
 Tensor(const std::string & name,                    //in: tensor name
        const Tensor & left_tensor,                  //in: left tensor
        const Tensor & right_tensor,                 //in: right tensor
        const std::vector<TensorLeg> & contraction); //in: tensor contraction pattern
 /** Create a tensor as a result of contraction of an isometric tensor
     with its conjugate over the isometric group of dimensions.
     The remaining tensor dimensions enter the destination tensor
     in order, first from the left tensor, then from the right tensor. **/
 Tensor(const std::string & name,             //in: tensor name
        const Tensor & isometric_tensor,      //in: isometric tensor
        std::vector<TensorLeg> & contraction, //out: tensor contraction pattern (see above)
        unsigned int iso_group_id = 0);       //in: isometric dimension group selector (0 or 1)
 /** Create a tensor from a byte packet. **/
 Tensor(BytePacket & byte_packet);

 /** Create a tensor by permuting another tensor. **/
 Tensor(const Tensor & another,                   //in: another tensor
        const std::vector<unsigned int> & order); //in: new order (N2O)

 Tensor(const Tensor & tensor) = default;
 Tensor & operator=(const Tensor & tensor) = default;
 Tensor(Tensor && tensor) noexcept = default;
 Tensor & operator=(Tensor && tensor) noexcept = default;
 virtual ~Tensor() = default;

 virtual std::shared_ptr<Tensor> clone() const;

 virtual void pack(BytePacket & byte_packet) const override;
 virtual void unpack(BytePacket & byte_packet) override;

 virtual bool isComposite() const;

 /** Returns TRUE if the tensor is congruent to another tensor and
     it is also decomposed (or replicated) in the same way.
     By being decomposed in the same way, the tensor is meant to
     have exactly the same subset of dimensions split in exactly
     the same way in exactly the same order. **/
 virtual bool isConformantTo(const Tensor & another) const;

 /** Print. **/
 virtual void printIt(bool with_hash = false) const;
 virtual void printItFile(std::ofstream & output_file,
                          bool with_hash = false) const;

 /** Renames the tensor. Do not use this method after the tensor has been allocated
     storage as it will mess up higher-level tensor allocation map! **/
 virtual void rename(const std::string & name);
 virtual void rename(); //a unique tensor name will be generated automatically via tensor hash

 /** Get tensor name. **/
 const std::string & getName() const;

 /** Get the tensor rank (order). **/
 unsigned int getRank() const;

 /** Get the tensor volume (number of elements). **/
 std::size_t getVolume() const;

 /** Get the tensor size (bytes). **/
 std::size_t getSize() const;

 /** Get the tensor shape. **/
 const TensorShape & getShape() const;

 /** Get the tensor signature. **/
 const TensorSignature & getSignature() const;

 /** Get the extent of a specific tensor dimension. **/
 DimExtent getDimExtent(unsigned int dim_id) const;

 /** Get the extents of all tensor dimensions. **/
 const std::vector<DimExtent> & getDimExtents() const;

 /** Get the strides for all tensor dimensions.
     Column-major tensor storage layout is assumed. **/
 const std::vector<DimExtent> getDimStrides(DimExtent * volume = nullptr) const;

 /** Get the space/subspace id for a specific tensor dimension. **/
 SpaceId getDimSpaceId(unsigned int dim_id) const;
 SubspaceId getDimSubspaceId(unsigned int dim_id) const;
 std::pair<SpaceId,SubspaceId> getDimSpaceAttr(unsigned int dim_id) const;

 /** Returns TRUE if the tensor is congruent to another tensor, that is,
     it has the same shape and signature. **/
 bool isCongruentTo(const Tensor & another) const;

 /** Deletes a specific tensor dimension, reducing the tensor rank by one. **/
 void deleteDimension(unsigned int dim_id);

 /** Appends a new dimension to the tensor at the end, increasing the tensor rank by one. **/
 void appendDimension(std::pair<SpaceId,SubspaceId> subspace,
                      DimExtent dim_extent);
 void appendDimension(DimExtent dim_extent);

 /** Replaces a tensor dimension. **/
 void replaceDimension(unsigned int dim_id,
                       std::pair<SpaceId,SubspaceId> subspace,
                       DimExtent dim_extent);
 void replaceDimension(unsigned int dim_id,
                       std::pair<SpaceId,SubspaceId> subspace);
 void replaceDimension(unsigned int dim_id,
                       DimExtent dim_extent);

 /** Creates a new tensor from the current tensor by selecting a subset of its modes.
     Vector mode_mask must have the size equal to the original tensor rank:
     mode_mask[i] == mask_val will select dimension i for appending to the subtensor. **/
 std::shared_ptr<Tensor> createSubtensor(const std::string & name,           //in: subtensor name
                                         const std::vector<int> & mode_mask, //in: mode masks
                                         int mask_val) const;                //in: chosen mask value

 /** Creates a related tensor from a given tensor by updating its subspaces and dimension extents. **/
 std::shared_ptr<Tensor> createSubtensor(const std::vector<SubspaceId> & subspaces,         //in: new defining subspaces
                                         const std::vector<DimExtent> & dim_extents) const; //in: new dimension extents

 /** Generates subtensors from a given tensor by splitting one of its dimensions. **/
 std::vector<std::shared_ptr<Tensor>> createSubtensors(unsigned int dim_id,
                                                       DimExtent num_segments = 2) const;

 /** Sets the tensor element type. **/
 void setElementType(TensorElementType element_type);

 /** Returns the tensor element type. **/
 TensorElementType getElementType() const;

 /** Registers an isometry in the tensor. **/
 void registerIsometry(const std::vector<unsigned int> & isometry);

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

 /** Returns a unique integer hash for the tensor object. **/
 TensorHashType getTensorHash() const;

protected:

 std::string name_;               //tensor name
 TensorShape shape_;              //tensor shape
 TensorSignature signature_;      //tensor signature
 TensorElementType element_type_; //tensor element type (optional)
 std::list<std::vector<unsigned int>> isometries_; //available isometries (optional)
};


//FREE FUNCTIONS:

/** Returns the size of a tensor element type in bytes. **/
inline std::size_t tensor_element_type_size(TensorElementType tensor_element_type)
{
 switch(tensor_element_type){
  case(TensorElementType::REAL16): return TensorElementTypeSize<TensorElementType::REAL16>();
  case(TensorElementType::REAL32): return TensorElementTypeSize<TensorElementType::REAL32>();
  case(TensorElementType::REAL64): return TensorElementTypeSize<TensorElementType::REAL64>();
  case(TensorElementType::COMPLEX16): return TensorElementTypeSize<TensorElementType::COMPLEX16>();
  case(TensorElementType::COMPLEX32): return TensorElementTypeSize<TensorElementType::COMPLEX32>();
  case(TensorElementType::COMPLEX64): return TensorElementTypeSize<TensorElementType::COMPLEX64>();
 }
 return TensorElementTypeSize<TensorElementType::VOID>();
}

/** Generates a unique name for a given tensor. **/
std::string generateTensorName(const Tensor & tensor,       //in: tensor stored on heap
                               const std::string & prefix); //in: desired name prefix

/** Compares a specific dimension of two tensors. **/
inline bool tensor_dims_conform(const Tensor & tensor1,
                                const Tensor & tensor2,
                                unsigned int dim1,
                                unsigned int dim2)
{
 return (tensor1.getDimSpaceAttr(dim1) == tensor2.getDimSpaceAttr(dim2)
      && tensor1.getDimExtent(dim1) == tensor2.getDimExtent(dim2));
}


//TEMPLATES:
template<typename T>
Tensor::Tensor(const std::string & name,
               std::initializer_list<T> extents,
               std::initializer_list<std::pair<SpaceId,SubspaceId>> subspaces):
name_(name), shape_(extents), signature_(subspaces), element_type_(TensorElementType::VOID)
{
 //DEBUG:
 if(signature_.getRank() != shape_.getRank()) std::cout << "ERROR(Tensor::Tensor): Signature/Shape size mismatch!" << std::endl;
 assert(signature_.getRank() == shape_.getRank());
}

template<typename T>
Tensor::Tensor(const std::string & name,
               const std::vector<T> & extents,
               const std::vector<std::pair<SpaceId,SubspaceId>> & subspaces):
name_(name), shape_(extents), signature_(subspaces), element_type_(TensorElementType::VOID)
{
 //DEBUG:
 if(signature_.getRank() != shape_.getRank()) std::cout << "ERROR(Tensor::Tensor): Signature/Shape size mismatch!" << std::endl;
 assert(signature_.getRank() == shape_.getRank());
}

template<typename T>
Tensor::Tensor(const std::string & name,
               std::initializer_list<T> extents):
name_(name), shape_(extents), signature_(static_cast<unsigned int>(extents.size())), element_type_(TensorElementType::VOID)
{
}

template<typename T>
Tensor::Tensor(const std::string & name,
               const std::vector<T> & extents):
name_(name), shape_(extents), signature_(static_cast<unsigned int>(extents.size())), element_type_(TensorElementType::VOID)
{
}

} //namespace numerics

/** Creates a new Tensor as a shared pointer. **/
template<typename... Args>
inline std::shared_ptr<numerics::Tensor> makeSharedTensor(Args&&... args)
{
 return std::make_shared<numerics::Tensor>(std::forward<Args>(args)...);
}

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_HPP_
