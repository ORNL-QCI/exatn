/** ExaTN::Numerics: Tensor
REVISION: 2018/11/16

Copyright (C) 2018-2018 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2018 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include "tensor_basic.hpp"
#include "tensor_signature.hpp"
#include "tensor_shape.hpp"

#include <assert.h>

#include <iostream>
#include <type_traits>
#include <string>
#include <initializer_list>
#include <vector>

namespace exatn{

namespace numerics{

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

 Tensor(const Tensor & tensor) = default;
 Tensor & operator=(const Tensor & tensor) = default;
 Tensor(Tensor && tensor) = default;
 Tensor & operator=(Tensor && tensor) = default;
 virtual ~Tensor() = default;

 /** Print. **/
 void printIt() const;

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

private:

 std::string name_;
 TensorShape shape_;
 TensorSignature signature_;
};


//TEMPLATES:
template<typename T>
Tensor::Tensor(const std::string & name,
               std::initializer_list<T> extents,
               std::initializer_list<std::pair<SpaceId,SubspaceId>> subspaces):
name_(name), shape_(extents), signature_(subspaces)
{
 if(signature_.getRank() != shape_.getRank()) std::cout << "ERROR(Tensor::Tensor): Signature/Shape size mismatch!" << std::endl;
 assert(signature_.getRank() == shape_.getRank());
}

template<typename T>
Tensor::Tensor(const std::string & name,
               const std::vector<T> & extents,
               const std::vector<std::pair<SpaceId,SubspaceId>> & subspaces):
name_(name), shape_(extents), signature_(subspaces)
{
 if(signature_.getRank() != shape_.getRank()) std::cout << "ERROR(Tensor::Tensor): Signature/Shape size mismatch!" << std::endl;
 assert(signature_.getRank() == shape_.getRank());
}

template<typename T>
Tensor::Tensor(const std::string & name,
               std::initializer_list<T> extents):
name_(name), shape_(extents), signature_(extents.size(),std::pair<SpaceId,SubspaceId>(SOME_SPACE,0))
{
}

template<typename T>
Tensor::Tensor::Tensor(const std::string & name,
                       const std::vector<T> & extents):
name_(name), shape_(extents), signature_(extents.size(),std::pair<SpaceId,SubspaceId>(SOME_SPACE,0))
{
}

} //namespace numerics

} //namespace exatn

#endif //TENSOR_HPP_
