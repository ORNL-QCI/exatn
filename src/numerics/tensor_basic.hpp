/** ExaTN: Tensor basic types and parameters
REVISION: 2018/10/31

Copyright (C) 2018-2018 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2018 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef TENSOR_BASIC_HPP_
#define TENSOR_BASIC_HPP_

#include <cstdint>

namespace exatn{

using Int4 = int32_t;
using Int8 = int64_t;
using UInt4 = uint32_t;
using UInt8 = uint64_t;

using SpaceId = unsigned int;              //space id type
using SubspaceId = unsigned long long int; //subspace id type
using DimExtent = unsigned long long int;  //dimension extent type
using DimOffset = unsigned long long int;  //dimension base offset type

const SpaceId SOME_SPACE = 0; //any unregistered space (all registered spaces will have SpaceId > 0)

} //namespace exatn

#endif //TENSOR_BASIC_HPP_
