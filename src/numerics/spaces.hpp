/** ExaTN::Numerics: Spaces/Subspaces
REVISION: 2019/01/27

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef SPACES_HPP_
#define SPACES_HPP_

#include "tensor_basic.hpp"
#include "space_basis.hpp"

#include <assert.h>

#include <iostream>
#include <string>
#include <unordered_map>

namespace exatn{

namespace numerics{

class SpaceRegister{
public:

 SpaceRegister();
 SpaceRegister(const SpaceRegister & space_register) = delete;
 SpaceRegister & operator=(const SpaceRegister & space_register) = delete;
 SpaceRegister(SpaceRegister && space_register) = default;
 SpaceRegister & operator=(SpaceRegister && space_register) = default;
 virtual ~SpaceRegister() = default;

 /** Print. **/
 void printIt() const;

 /** Registers a vector space by providing only its dimension and optional name. **/
 SpaceId registerSpace(DimExtent space_dim, const std::string & space_name = std:string());

 /** Registers a vector space by providing its concrete basis and optional name. **/
 SpaceId registerSpace(const SpaceBasis & basis, const std::string & space_name = std:string());

 /** Returns a reference to the registered vector space. **/
 const VectorSpace & getSpace(SpaceId space_id);
 const VectorSpace & getSpace(const std::string & space_name);

private:

 SpaceId last_; //last registered Space ID
 std::unordered_map<SpaceId,VectorSpace> space_register_;
 std::unordered_map<std::string,SpaceId> space_name2id_;
};

} //namespace numerics

} //namespace exatn

#endif //SPACES_HPP_
