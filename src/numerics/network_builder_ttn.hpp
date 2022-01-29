/** ExaTN::Numerics: Tensor network builder: Tree: Tree Tensor Network
REVISION: 2022/01/29

Copyright (C) 2018-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2022 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Builds a tree tensor network.
     Parameters:
     * max_bond_dim: >0: Maximal internal bond dimension;
     * arity: >1: Tree arity;
     * isometric: {0/1}: Whether or not all tensor factors except the root are isometric;
 (b) Leg numeration (tensor network vector):

    [0] [1]      [2] [3] ...
     0   1        0   1
      \ /          \ /
       X            X
        \          /
         2        2
         \       /
          \     /
           0   1
            \ /
             X
              \
               2
                \
                   ...

 (c) Leg numeration (tensor network operator): Each tensor X in the layer
     of tree leaves gets two more legs (#3,#4) for the bra space (that is,
     legs #0 and #1 are mapped to legs #3 and #4). The bra legs will be
     appended to the end of the output tensor of the tensor network.

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
 int isometric_;          //isometry
};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_NETWORK_BUILDER_TTN_HPP_
