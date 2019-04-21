/** ExaTN::Numerics: Factory
REVISION: 2019/04/20

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef NUMERICS_FACTORY_HPP_
#define NUMERICS_FACTORY_HPP_

#include "tensor_basic.hpp"
#include "register.hpp"
#include "tensor_factory.hpp"

#include <string>
#include <unordered_map>

namespace exatn{

namespace numerics{

class Factory{ //singleton

public:

 Factory() = default;
 Factory(const Factory &) = delete;
 Factory & operator=(const Factory &) = delete;
 Factory(Factory &&) = default;
 Factory & operator=(Factory &&) = default;
 ~Factory() = default;

private:

 SpaceRegister space_register_; //register of vector spaces and their named subspaces
 std::unordered_map<std::string,SpaceId> name2id_; //maps a subspace name to its parental vector space id

};

} //namespace numerics

} //namespace exatn

#endif //NUMERICS_FACTORY_HPP_
