/** ExaTN::Numerics: Tensor network builder: Tree: Tree Tensor Network
REVISION: 2019/11/01

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
**/

#ifndef EXATN_NUMERICS_NETWORK_BUILDER_TREE_HPP_
#define EXATN_NUMERICS_NETWORK_BUILDER_TREE_HPP_

#include "tensor_basic.hpp"
#include "network_builder.hpp"

#include <string>
#include <memory>

#include "errors.hpp"

namespace exatn{

namespace numerics{

class NetworkBuilderTree: public NetworkBuilder{

public:

 NetworkBuilderTree();
 NetworkBuilderTree(const NetworkBuilderTree &) = default;
 NetworkBuilderTree & operator=(const NetworkBuilderTree &) = default;
 NetworkBuilderTree(NetworkBuilderTree &&) noexcept = default;
 NetworkBuilderTree & operator=(NetworkBuilderTree &&) noexcept = default;
 virtual ~NetworkBuilderTree() = default;

 /** Retrieves a specific parameter of the tensor network builder. **/
 virtual bool getParameter(const std::string & name, long long * value) const override;

 /** Sets a specific parameter of the tensor network builder. **/
 virtual bool setParameter(const std::string & name, long long value) override;

 /** Builds a tensor network of a specific kind. **/
 virtual void build(TensorNetwork & network) override;

 static std::unique_ptr<NetworkBuilder> createNew();

private:

 long long max_bond_dim_; //maximal internal bond dimension
 long long arity_;        //tree arity
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_NETWORK_BUILDER_TREE_HPP_
