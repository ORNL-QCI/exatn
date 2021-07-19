/** ExaTN::Numerics: Tensor network builder
REVISION: 2021/06/25

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) A tensor network builder allows building complex tensor networks of a specific kind.
**/

#ifndef EXATN_NUMERICS_NETWORK_BUILDER_HPP_
#define EXATN_NUMERICS_NETWORK_BUILDER_HPP_

#include "tensor_basic.hpp"

#include <string>
#include <memory>

#include "errors.hpp"

namespace exatn{

namespace numerics{

class TensorNetwork;

class NetworkBuilder{ //abstract

public:

 NetworkBuilder() = default;
 NetworkBuilder(const NetworkBuilder &) = default;
 NetworkBuilder & operator=(const NetworkBuilder &) = default;
 NetworkBuilder(NetworkBuilder &&) noexcept = default;
 NetworkBuilder & operator=(NetworkBuilder &&) noexcept = default;
 virtual ~NetworkBuilder() = default;

 /** Retrieves a specific parameter of the tensor network builder. **/
 virtual bool getParameter(const std::string & name, long long * value) const = 0;

 /** Sets a specific parameter of the tensor network builder. **/
 virtual bool setParameter(const std::string & name, long long value) = 0;

 /** Builds a tensor network of a specific kind. On input, the tensor
     network must only contain the output tensor with dummy legs.
     If tensor_operator = TRUE, the tensor network operator will
     be built instead of the tensor network vector. In that case,
     the first half legs correspond to ket while the rest to bra. **/
 virtual void build(TensorNetwork & network,           //inout: tensor network
                    bool tensor_operator = false) = 0; //in: tensor network vector or operator

};

using createNetworkBuilderFn = std::unique_ptr<NetworkBuilder> (*)(void);

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_NETWORK_BUILDER_HPP_
