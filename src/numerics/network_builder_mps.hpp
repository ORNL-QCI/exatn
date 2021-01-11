/** ExaTN::Numerics: Tensor network builder: MPS: Matrix Product State
REVISION: 2021/01/11

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Builds a matrix product state tensor network:
     Parameters:
     * max_bond_dim: Maximal internal bond dimension;
**/

#ifndef EXATN_NUMERICS_NETWORK_BUILDER_MPS_HPP_
#define EXATN_NUMERICS_NETWORK_BUILDER_MPS_HPP_

#include "tensor_basic.hpp"
#include "network_builder.hpp"

#include <string>
#include <memory>

#include "errors.hpp"

namespace exatn{

namespace numerics{

class NetworkBuilderMPS: public NetworkBuilder{

public:

 NetworkBuilderMPS();
 NetworkBuilderMPS(const NetworkBuilderMPS &) = default;
 NetworkBuilderMPS & operator=(const NetworkBuilderMPS &) = default;
 NetworkBuilderMPS(NetworkBuilderMPS &&) noexcept = default;
 NetworkBuilderMPS & operator=(NetworkBuilderMPS &&) noexcept = default;
 virtual ~NetworkBuilderMPS() = default;

 /** Retrieves a specific parameter of the tensor network builder. **/
 virtual bool getParameter(const std::string & name, long long * value) const override;

 /** Sets a specific parameter of the tensor network builder. **/
 virtual bool setParameter(const std::string & name, long long value) override;

 /** Builds a tensor network of a specific kind. On entrance, the passed
     tensor network must only have the output tensor. On exit, it will be
     the fully constructed tensor network of a specific kind. **/
 virtual void build(TensorNetwork & network) override; //inout: tensor network

 static std::unique_ptr<NetworkBuilder> createNew();

private:

 long long max_bond_dim_; //maximal internal bond dimension
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_NETWORK_BUILDER_MPS_HPP_
