/** ExaTN::Numerics: Factories
REVISION: 2019/03/17

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef NUMERICS_FACTORY_HPP_
#define NUMERICS_FACTORY_HPP_

#include "tensor_basic.hpp"
#include "register.hpp"

#include <string>
#include <unordered_map>

namespace exatn{

namespace numerics{

class Factories{ //singleton

public:

 Factories() = default;
 Factories(const Factories &) = delete;
 Factories & operator=(const Factories &) = delete;
 Factories(const Factories &&) = default;
 Factories & operator=(const Factories &&) = default;
 ~Factories() = default;

private:

 SpaceRegister space_register_; //register of vector spaces and their named subspaces
 std::unordered_map<std::string,SpaceId> name2id_; //maps a subspace name to its parental vector space id

};

} //namespace numerics

} //namespace exatn

#endif //NUMERICS_FACTORY_HPP_
