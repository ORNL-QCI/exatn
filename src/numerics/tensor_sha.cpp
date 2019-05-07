/** ExaTN::Numerics: TAL-SH Tensor
REVISION: 2019/05/07

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_sha.hpp"

#include <assert.h>

namespace exatn{

namespace numerics{

TensorSha::TensorSha(const std::string & name,
                     const TensorShape & shape,
                     const TensorSignature & signature):
Tensor(name,shape,signature)
{
}

TensorSha::TensorSha(const std::string & name,
                     const TensorShape & shape):
Tensor(name,shape)
{
}

} //namespace numerics

} //namespace exatn
