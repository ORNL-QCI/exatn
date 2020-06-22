/** ExaTN: Tensor basic types and parameters
REVISION: 2020/06/22

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef EXATN_NUMERICS_TENSOR_BASIC_HPP_
#define EXATN_NUMERICS_TENSOR_BASIC_HPP_

#include <complex>

#include <cstdint>

namespace exatn{

using Int4 = int32_t;
using Int8 = int64_t;
using UInt4 = uint32_t;
using UInt8 = uint64_t;

using SpaceId = unsigned int;              //space id type
using SubspaceId = unsigned long long int; //subspace id type
using SymmetryId = int;                    //symmetry id type
using DimExtent = unsigned long long int;  //dimension extent type
using DimOffset = unsigned long long int;  //dimension base offset type

using ScopeId = unsigned int; //TAProL scope ID type

constexpr DimExtent MAX_SPACE_DIM = 0xFFFFFFFFFFFFFFFF; //max dimension of unregistered (anonymous) spaces
constexpr SpaceId SOME_SPACE = 0; //any unregistered (anonymous) space (all registered spaces will have SpaceId > 0)
constexpr SubspaceId FULL_SUBSPACE = 0; //every space has its trivial (full) subspace automatically registered as subspace 0
constexpr SubspaceId UNREG_SUBSPACE = 0xFFFFFFFFFFFFFFFF; //id of any unregistered subspace

enum class LegDirection{
 UNDIRECT, //no direction
 INWARD,   //inward direction
 OUTWARD   //outward direction
};

enum class TensorOpCode{
 NOOP,              //no operation
 CREATE,            //tensor creation
 DESTROY,           //tensor destruction
 TRANSFORM,         //tensor transformation/initialization
 SLICE,             //tensor slicing
 INSERT,            //tensor insertion
 ADD,               //tensor addition
 CONTRACT,          //tensor contraction
 DECOMPOSE_SVD3,    //tensor decomposition via SVD into three tensor factors
 DECOMPOSE_SVD2,    //tensor decomposition via SVD into two tensor factors
 ORTHOGONALIZE_SVD, //tensor orthogonalization via SVD
 ORTHOGONALIZE_MGS, //tensor orthogonalization via Modified Gram-Schmidt
 BROADCAST,         //tensor broadcast (parallel execution only)
 ALLREDUCE          //tensor allreduce (parallel execution only)
};

enum class TensorElementType{
 VOID,
 REAL16,
 REAL32,
 REAL64,
 COMPLEX16,
 COMPLEX32,
 COMPLEX64
};

//TensorElementTypeSize<enum TensorElementType>() --> Size in bytes:
template <TensorElementType> constexpr std::size_t TensorElementTypeSize();
template <> constexpr std::size_t TensorElementTypeSize<TensorElementType::VOID>(){return 0;} //0 bytes
template <> constexpr std::size_t TensorElementTypeSize<TensorElementType::REAL16>(){return 2;} //2 bytes
template <> constexpr std::size_t TensorElementTypeSize<TensorElementType::REAL32>(){return 4;} //4 bytes
template <> constexpr std::size_t TensorElementTypeSize<TensorElementType::REAL64>(){return 8;} //8 bytes
template <> constexpr std::size_t TensorElementTypeSize<TensorElementType::COMPLEX16>(){return 4;} //4 bytes
template <> constexpr std::size_t TensorElementTypeSize<TensorElementType::COMPLEX32>(){return 8;} //8 bytes
template <> constexpr std::size_t TensorElementTypeSize<TensorElementType::COMPLEX64>(){return 16;} //16 bytes

//TensorDataType<enum TensorElementType>::value --> C++ type:
template <TensorElementType> struct TensorDataType{
 using value = void;
};
template <> struct TensorDataType<TensorElementType::REAL32>{
 using value = float;
 static constexpr const value ZERO {0.0f};
 static constexpr const value UNITY {1.0f};
 static constexpr std::size_t size() {return sizeof(value);}
};
template <> struct TensorDataType<TensorElementType::REAL64>{
 using value = double;
 static constexpr const value ZERO {0.0};
 static constexpr const value UNITY {1.0};
 static constexpr std::size_t size() {return sizeof(value);}
};
template <> struct TensorDataType<TensorElementType::COMPLEX32>{
 using value = std::complex<float>;
 static constexpr const value ZERO {0.0f,0.0f};
 static constexpr const value UNITY {1.0f,0.0f};
 static constexpr std::size_t size() {return sizeof(value);}
};
template <> struct TensorDataType<TensorElementType::COMPLEX64>{
 using value = std::complex<double>;
 static constexpr const value ZERO {0.0,0.0};
 static constexpr const value UNITY {1.0,0.0};
 static constexpr std::size_t size() {return sizeof(value);}
};

//TensorDataKind<C++ type>::value --> enum TensorElementType:
template <typename T> struct TensorDataKind{
 static constexpr const TensorElementType value = TensorElementType::VOID;
};
template <> struct TensorDataKind<float>{
 static constexpr const TensorElementType value = TensorElementType::REAL32;
 static constexpr const float ZERO {0.0f};
 static constexpr const float UNITY {1.0f};
 static constexpr std::size_t size() {return sizeof(float);}
};
template <> struct TensorDataKind<double>{
 static constexpr const TensorElementType value = TensorElementType::REAL64;
 static constexpr const double ZERO {0.0};
 static constexpr const double UNITY {1.0};
 static constexpr std::size_t size() {return sizeof(double);}
};
template <> struct TensorDataKind<std::complex<float>>{
 static constexpr const TensorElementType value = TensorElementType::COMPLEX32;
 static constexpr const std::complex<float> ZERO {0.0f,0.0f};
 static constexpr const std::complex<float> UNITY {1.0f,0.0f};
 static constexpr std::size_t size() {return sizeof(std::complex<std::complex<float>>);}
};
template <> struct TensorDataKind<std::complex<double>>{
 static constexpr const TensorElementType value = TensorElementType::COMPLEX64;
 static constexpr const std::complex<double> ZERO {0.0,0.0};
 static constexpr const std::complex<double> UNITY {1.0,0.0};
 static constexpr std::size_t size() {return sizeof(std::complex<std::complex<double>>);}
};

} //namespace exatn

#endif //EXATN_NUMERICS_TENSOR_BASIC_HPP_
