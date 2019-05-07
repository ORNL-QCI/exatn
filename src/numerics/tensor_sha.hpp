/** ExaTN::Numerics: TAL-SH Tensor
REVISION: 2019/05/07

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef EXATN_NUMERICS_TENSOR_SHA_HPP_
#define EXATN_NUMERICS_TENSOR_SHA_HPP_

#include "tensor_basic.hpp"
#include "tensor.hpp"

//#include "talshxx.hpp"

#include <assert.h>

#include <iostream>
#include <type_traits>
#include <string>
#include <initializer_list>
#include <vector>

namespace exatn{

namespace numerics{

class TensorSha: public Tensor{
public:

 /** Create a tensor by providing its name, shape and signature. **/
 TensorSha(const std::string & name,           //tensor name
           const TensorShape & shape,          //tensor shape
           const TensorSignature & signature); //tensor signature
 /** Create a tensor by providing its name and shape.
     The signature defaults to SOME_SPACE spaces and lbound=0 subspaces. **/
 TensorSha(const std::string & name,           //tensor name
           const TensorShape & shape);         //tensor shape
 /** Create a tensor by providing its name, shape and signature from scratch. **/
 template<typename T>
 TensorSha(const std::string & name,                                        //tensor name
           std::initializer_list<T> extents,                                //tensor dimension extents
           std::initializer_list<std::pair<SpaceId,SubspaceId>> subspaces); //tensor dimension defining subspaces
 template<typename T>
 TensorSha(const std::string & name,                                      //tensor name
           const std::vector<T> & extents,                                //tensor dimension extents
           const std::vector<std::pair<SpaceId,SubspaceId>> & subspaces); //tensor dimension defining subspaces
 /** Create a tensor by providing its name and shape from scratch.
     The signature defaults to SOME_SPACE spaces and lbound=0 subspaces. **/
 template<typename T>
 TensorSha(const std::string & name,          //tensor name
           std::initializer_list<T> extents); //tensor dimension extents
 template<typename T>
 TensorSha(const std::string & name,        //tensor name
           const std::vector<T> & extents); //tensor dimension extents

 TensorSha(const TensorSha & tensor) = default;
 TensorSha & operator=(const TensorSha & tensor) = default;
 TensorSha(TensorSha && tensor) noexcept = default;
 TensorSha & operator=(TensorSha && tensor) noexcept = default;
 virtual ~TensorSha() = default;

private:

 //talsh::Tensor tensor_; //TAL-SH tensor
};


//TEMPLATES:
template<typename T>
TensorSha::TensorSha(const std::string & name,
                     std::initializer_list<T> extents,
                     std::initializer_list<std::pair<SpaceId,SubspaceId>> subspaces):
Tensor(name,extents,subspaces)
{
}

template<typename T>
TensorSha::TensorSha(const std::string & name,
                     const std::vector<T> & extents,
                     const std::vector<std::pair<SpaceId,SubspaceId>> & subspaces):
Tensor(name,extents,subspaces)
{
}

template<typename T>
TensorSha::TensorSha(const std::string & name,
                     std::initializer_list<T> extents):
Tensor(name,extents)
{
}

template<typename T>
TensorSha::TensorSha(const std::string & name,
                     const std::vector<T> & extents):
Tensor(name,extents)
{
}

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_SHA_HPP_
