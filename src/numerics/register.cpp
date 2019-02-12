/** ExaTN::Numerics: Register
REVISION: 2019/02/12

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "register.hpp"

#include <iostream>

#include "assert.h"

namespace exatn{

namespace numerics{

SubspaceRegEntry::SubspaceRegEntry(std::shared_ptr<Subspace> & subspace):
 subspace_(subspace)
{
}

SubspaceRegEntry::SubspaceRegEntry(std::shared_ptr<Subspace> && subspace):
 subspace_(subspace)
{
}


SubspaceId SubspaceRegister::registerSubspace(std::shared_ptr<Subspace> & subspace)
{
 assert(subspace->getRegisteredId() == 0);
 SubspaceId id = subspaces_.size();
 subspace->resetRegisteredId(id);
 subspaces_.emplace_back(SubspaceRegEntry(subspace));
 return id;
}

SubspaceId SubspaceRegister::registerSubspace(std::shared_ptr<Subspace> && subspace)
{
 assert(subspace->getRegisteredId() == 0);
 SubspaceId id = subspaces_.size();
 subspace->resetRegisteredId(id);
 subspaces_.emplace_back(SubspaceRegEntry(subspace));
 return id;
}


SpaceRegEntry::SpaceRegEntry(std::shared_ptr<VectorSpace> & space):
 space_(space)
{
 DimOffset lower = 0;
 DimOffset upper = lower + space_->getSpaceDimension() - 1;
 const std::string & space_name = space_->getSpaceName();
 SubspaceId id = subspaces_.registerSubspace(std::make_shared<Subspace>(space_.get(),lower,upper,space_name)); //register the full space as its trivial subspace under the same name
 assert(id == 0);
}

SpaceRegEntry::SpaceRegEntry(std::shared_ptr<VectorSpace> && space):
 space_(space)
{
 DimOffset lower = 0;
 DimOffset upper = lower + space_->getSpaceDimension() - 1;
 const std::string & space_name = space_->getSpaceName();
 SubspaceId id = subspaces_.registerSubspace(std::make_shared<Subspace>(space_.get(),lower,upper,space_name)); //register the full space as its trivial subspace under the same name
 assert(id == 0);
}


SpaceRegister::SpaceRegister()
{
 spaces_.emplace_back(SpaceRegEntry(std::make_shared<VectorSpace>(MAX_SPACE_DIM))); //Space 0 is an unnamed abstract space of maximal dimension
}


SpaceId SpaceRegister::registerSpace(std::shared_ptr<VectorSpace> & space)
{
 assert(space->getRegisteredId() == SOME_SPACE);
 SpaceId id = spaces_.size();
 space->resetRegisteredId(id);
 spaces_.emplace_back(SpaceRegEntry(space));
 return id;
}

} //namespace numerics

} //namespace exatn
