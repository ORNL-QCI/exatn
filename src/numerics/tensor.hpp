/** ExaTN::Numerics: Abstract Tensor
REVISION: 2019/07/22

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** NOTES:
 Tensor specification requires:
 (a) Symbolic tensor name;
 (b) Tensor rank (number of tensor dimensions) and
     tensor shape (extents of all tensor dimensions);
 (c) Optional tensor signature (space/subspace identifier for all tensor dimensions).

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
#include "tensor_shape.hpp"
#include "tensor_signature.hpp"
#include "tensor_leg.hpp"

#include <assert.h>

#include <iostream>
#include <type_traits>
#include <string>
#include <initializer_list>
#include <vector>

namespace exatn{

namespace numerics{

using TensorHashType = std::size_t;

class Tensor{
public:

 /** Create a tensor by providing its name, shape and signature. **/
 Tensor(const std::string & name,           //tensor name
        const TensorShape & shape,          //tensor shape
        const TensorSignature & signature); //tensor signature
 /** Create a tensor by providing its name and shape.
     The signature defaults to SOME_SPACE spaces and lbound=0 subspaces. **/
 Tensor(const std::string & name,           //tensor name
        const TensorShape & shape);         //tensor shape
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
      contraction.size() = left_rank + right_rank;
      Output tensor id = 0;
      Left input tensor id = 1;
      Right input tensor id = 2. **/
 Tensor(const std::string & name,                    //tensor name
        const Tensor & left_tensor,                  //left tensor
        const Tensor & right_tensor,                 //right tensor
        const std::vector<TensorLeg> & contraction); //tensor contraction pattern

 Tensor(const Tensor & tensor) = default;
 Tensor & operator=(const Tensor & tensor) = default;
 Tensor(Tensor && tensor) noexcept = default;
 Tensor & operator=(Tensor && tensor) noexcept = default;
 virtual ~Tensor() = default;

 /** Print. **/
 void printIt() const;

 /** Get tensor name. **/
 const std::string & getName() const;
 /** Get the tensor rank (order). **/
 unsigned int getRank() const;
 /** Get the tensor shape. **/
 const TensorShape & getShape() const;
 /** Get the tensor signature. **/
 const TensorSignature & getSignature() const;

 /** Get the extent of a specific tensor dimension. **/
 DimExtent getDimExtent(unsigned int dim_id) const;
 /** Get the extents of all tensor dimensions. **/
 const std::vector<DimExtent> & getDimExtents() const;

 /** Get the space/subspace id for a specific tensor dimension. **/
 SpaceId getDimSpaceId(unsigned int dim_id) const;
 SubspaceId getDimSubspaceId(unsigned int dim_id) const;
 std::pair<SpaceId,SubspaceId> getDimSpaceAttr(unsigned int dim_id) const;

 /** Deletes a specific tensor dimension, reducing the tensor rank by one. **/
 void deleteDimension(unsigned int dim_id);

 /** Appends a new dimension to the tensor at the end, increasing the tensor rank by one. **/
 void appendDimension(std::pair<SpaceId,SubspaceId> subspace,
                      DimExtent dim_extent);
 void appendDimension(DimExtent dim_extent);

 /** Get the unique integer tensor id. **/
 TensorHashType getTensorHash() const;

private:

 std::string name_;          //tensor name
 TensorShape shape_;         //tensor shape
 TensorSignature signature_; //tensor signature
};


//TEMPLATES:
template<typename T>
Tensor::Tensor(const std::string & name,
               std::initializer_list<T> extents,
               std::initializer_list<std::pair<SpaceId,SubspaceId>> subspaces):
name_(name), shape_(extents), signature_(subspaces)
{
 //DEBUG:
 if(signature_.getRank() != shape_.getRank()) std::cout << "ERROR(Tensor::Tensor): Signature/Shape size mismatch!" << std::endl;
 assert(signature_.getRank() == shape_.getRank());
}

template<typename T>
Tensor::Tensor(const std::string & name,
               const std::vector<T> & extents,
               const std::vector<std::pair<SpaceId,SubspaceId>> & subspaces):
name_(name), shape_(extents), signature_(subspaces)
{
 //DEBUG:
 if(signature_.getRank() != shape_.getRank()) std::cout << "ERROR(Tensor::Tensor): Signature/Shape size mismatch!" << std::endl;
 assert(signature_.getRank() == shape_.getRank());
}

template<typename T>
Tensor::Tensor(const std::string & name,
               std::initializer_list<T> extents):
name_(name), shape_(extents), signature_(static_cast<unsigned int>(extents.size()))
{
}

template<typename T>
Tensor::Tensor(const std::string & name,
               const std::vector<T> & extents):
name_(name), shape_(extents), signature_(static_cast<unsigned int>(extents.size()))
{
}

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_HPP_
