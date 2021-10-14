/** ExaTN::Numerics: Tensor network builder: MPS: Matrix Product State
REVISION: 2021/10/14

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Builds a matrix product state tensor network:
     Parameters:
     * max_bond_dim: Maximal internal bond dimension;
 (b) Leg numeration (tensor network vector):

     [0]    [1]    [2]         [n-2]  [n-1]
      0      1      1            1      1
      |      |      |            |      |
      X-1--0-X-2--0-X-2-- .. --0-X-2--0-X

 (c) Leg numeration (tensor network operator):

     [0]    [1]    [2]         [n-2]  [n-1]
      0      1      1            1      1
      |      |      |            |      |
      X-1--0-X-2--0-X-2-- .. --0-X-2--0-X
      |      |      |            |      |
      2      3      3            3      2
     [n]   [n+1]  [n+2]       [n*2-2][n*2-1]

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
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_NETWORK_BUILDER_MPS_HPP_
