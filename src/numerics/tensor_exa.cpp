/** ExaTN::Numerics: ExaTENSOR Tensor
REVISION: 2018/12/18

Copyright (C) 2018-2018 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2018 Oak Ridge National Laboratory (UT-Battelle) **/

#include "tensor_exa.hpp"

namespace exatn{

namespace numerics{

TensorExa::TensorExa(const std::string & name,
                     const TensorShape & shape,
                     const TensorSignature & signature):
Tensor(name,shape,signature)
{
}

TensorExa::TensorExa(const std::string & name,
                     const TensorShape & shape):
Tensor(name,shape)
{
}

} //namespace numerics

} //namespace exatn
