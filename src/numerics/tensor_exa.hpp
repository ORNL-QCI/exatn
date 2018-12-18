/** ExaTN::Numerics: ExaTENSOR Tensor
REVISION: 2018/12/18

Copyright (C) 2018-2018 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2018 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef TENSOR_EXA_HPP_
#define TENSOR_EXA_HPP_

#include "tensor_basic.hpp"
#include "tensor_signature.hpp"
#include "tensor_shape.hpp"
#include "tensor.hpp"

#include <assert.h>

#include <iostream>
#include <type_traits>
#include <string>
#include <initializer_list>
#include <vector>

namespace exatn{

namespace numerics{

class TensorExa: public Tensor{
public:

 /** Create a tensor by providing its name, shape and signature. **/
 TensorExa(const std::string & name,           //tensor name
           const TensorShape & shape,          //tensor shape
           const TensorSignature & signature); //tensor signature
 /** Create a tensor by providing its name and shape.
     The signature defaults to SOME_SPACE spaces and lbound=0 subspaces. **/
 TensorExa(const std::string & name,           //tensor name
           const TensorShape & shape);         //tensor shape
 /** Create a tensor by providing its name, shape and signature from scratch. **/
 template<typename T>
 TensorExa(const std::string & name,                                        //tensor name
           std::initializer_list<T> extents,                                //tensor dimension extents
           std::initializer_list<std::pair<SpaceId,SubspaceId>> subspaces); //tensor dimension defining subspaces
 template<typename T>
 TensorExa(const std::string & name,                                      //tensor name
           const std::vector<T> & extents,                                //tensor dimension extents
           const std::vector<std::pair<SpaceId,SubspaceId>> & subspaces); //tensor dimension defining subspaces
 /** Create a tensor by providing its name and shape from scratch.
     The signature defaults to SOME_SPACE spaces and lbound=0 subspaces. **/
 template<typename T>
 TensorExa(const std::string & name,          //tensor name
           std::initializer_list<T> extents); //tensor dimension extents
 template<typename T>
 TensorExa(const std::string & name,        //tensor name
           const std::vector<T> & extents); //tensor dimension extents

 TensorExa(const TensorExa & tensor) = default;
 TensorExa & operator=(const TensorExa & tensor) = default;
 TensorExa(TensorExa && tensor) = default;
 TensorExa & operator=(TensorExa && tensor) = default;
 virtual ~TensorExa() = default;

private:

 std::string exasymbol_; //symbolic representation of the tensor registered with ExaTENSOR
};


//TEMPLATES:
template<typename T>
TensorExa::TensorExa(const std::string & name,
                     std::initializer_list<T> extents,
                     std::initializer_list<std::pair<SpaceId,SubspaceId>> subspaces):
Tensor(name,extents,subspaces)
{
}

template<typename T>
TensorExa::TensorExa(const std::string & name,
                     const std::vector<T> & extents,
                     const std::vector<std::pair<SpaceId,SubspaceId>> & subspaces):
Tensor(name,extents,subspaces)
{
}

template<typename T>
TensorExa::TensorExa(const std::string & name,
                     std::initializer_list<T> extents):
Tensor(name,extents)
{
}

template<typename T>
TensorExa::TensorExa(const std::string & name,
                     const std::vector<T> & extents):
Tensor(name,extents)
{
}

} //namespace numerics

} //namespace exatn

#endif //TENSOR_EXA_HPP_
