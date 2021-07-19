/** ExaTN::Numerics: Tensor network builder: Tree: Tree Tensor Network
REVISION: 2021/06/25

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Builds a tree tensor network.
     Parameters:
     * max_bond_dim: Maximal internal bond dimension;
     * arity: Tree arity;
**/

#ifndef EXATN_NUMERICS_NETWORK_BUILDER_TTN_HPP_
#define EXATN_NUMERICS_NETWORK_BUILDER_TTN_HPP_

#include "tensor_basic.hpp"
#include "network_builder.hpp"

#include <string>
#include <memory>

#include "errors.hpp"

namespace exatn{

namespace numerics{

class NetworkBuilderTTN: public NetworkBuilder{

public:

 NetworkBuilderTTN();
 NetworkBuilderTTN(const NetworkBuilderTTN &) = default;
 NetworkBuilderTTN & operator=(const NetworkBuilderTTN &) = default;
 NetworkBuilderTTN(NetworkBuilderTTN &&) noexcept = default;
 NetworkBuilderTTN & operator=(NetworkBuilderTTN &&) noexcept = default;
 virtual ~NetworkBuilderTTN() = default;

 /** Retrieves a specific parameter of the tensor network builder. **/
 virtual bool getParameter(const std::string & name, long long * value) const override;

 /** Sets a specific parameter of the tensor network builder. **/
 virtual bool setParameter(const std::string & name, long long value) override;

 /** Builds a tensor network of a specific kind. On input, the tensor
     network must only contain the output tensor with dummy legs.
     If tensor_operator = TRUE, the tensor network operator will
     be built instead of the tensor network vector. In that case,
     the first half legs correspond to ket while the rest to bra. **/
 virtual void build(TensorNetwork & network,                //inout: tensor network
                    bool tensor_operator = false) override; //in: tensor network vector or operator

 static std::unique_ptr<NetworkBuilder> createNew();

private:

 long long max_bond_dim_; //maximal internal bond dimension
 long long arity_;        //tree arity
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_NETWORK_BUILDER_TTN_HPP_
