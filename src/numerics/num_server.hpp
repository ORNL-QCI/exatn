/** ExaTN::Numerics: Numerical server
REVISION: 2019/05/07

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef EXATN_NUMERICS_NUM_SERVER_HPP_
#define EXATN_NUMERICS_NUM_SERVER_HPP_

#include "tensor_basic.hpp"
#include "space_register.hpp"
#include "tensor_factory.hpp"
#include "tensor_network.hpp"

#include <string>
#include <unordered_map>

namespace exatn{

namespace numerics{

class NumServer{ //singleton

public:

 NumServer() = default;
 NumServer(const NumServer &) = delete;
 NumServer & operator=(const NumServer &) = delete;
 NumServer(NumServer &&) noexcept = default;
 NumServer & operator=(NumServer &&) noexcept = default;
 ~NumServer() = default;

private:

 SpaceRegister space_register_; //register of vector spaces and their named subspaces
 std::unordered_map<std::string,SpaceId> subname2id_; //maps a subspace name to its parental vector space id

};

} //namespace numerics

} //namespace exatn

#endif //EXATN_NUMERICS_NUM_SERVER_HPP_
