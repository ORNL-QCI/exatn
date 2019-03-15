/** ExaTN::Numerics: Register
REVISION: 2019/03/15

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
 SubspaceId id = subspace->getRegisteredId();
 if(id == UNREG_SUBSPACE){ //previously unregistered
  id = subspaces_.size(); //new registered subspace id (>0)
  bool unique = name2id_.insert({subspace->getName(),id}).second;
  if(unique){
   subspace->resetRegisteredId(id);
   subspaces_.emplace_back(SubspaceRegEntry(subspace)); //subspace register shares ownership of the stored subspace
  }else{
   std::cout << "WARNING: Attempt to register subspace with an already registered name: " << subspace->getName() << std::endl;
   assert(unique); //subspace with this name already exists, registration unsuccessful
  }
 }
 return id;
}

SubspaceId SubspaceRegister::registerSubspace(std::shared_ptr<Subspace> && subspace)
{
 SubspaceId id = subspace->getRegisteredId();
 if(id == UNREG_SUBSPACE){ //previously unregistered
  id = subspaces_.size(); //new registered subspace id (>0)
  bool unique = name2id_.insert({subspace->getName(),id}).second;
  if(unique){
   subspace->resetRegisteredId(id);
   subspaces_.emplace_back(SubspaceRegEntry(subspace)); //subspace register shares ownership of the stored subspace
  }else{
   std::cout << "WARNING: Attempt to register subspace with an already registered name: " << subspace->getName() << std::endl;
   assert(unique); //subspace with this name already exists, registration unsuccessful
  }
 }
 return id;
}


Subspace & SubspaceRegister::getSubspace(SubspaceId id) const
{
 assert(id < subspaces_.size());
 return *(subspaces_[id].subspace_);
}

Subspace & SubspaceRegister::getSubspace(const std::string & name) const
{
 auto it = name2id_.find(name); assert(it != name2id_.end());
 return *(subspaces_[it->second].subspace_);
}


SpaceRegEntry::SpaceRegEntry(std::shared_ptr<VectorSpace> & space):
 space_(space)
{
 DimOffset lower = 0;
 DimOffset upper = lower + space_->getDimension() - 1;
 const std::string & space_name = space_->getName();
 SubspaceId id = subspaces_.registerSubspace(std::make_shared<Subspace>(space_.get(),lower,upper,space_name)); //register the full space as its trivial subspace under the same name
 assert(id == FULL_SUBSPACE);
}

SpaceRegEntry::SpaceRegEntry(std::shared_ptr<VectorSpace> && space):
 space_(space)
{
 DimOffset lower = 0;
 DimOffset upper = lower + space_->getDimension() - 1;
 const std::string & space_name = space_->getName();
 SubspaceId id = subspaces_.registerSubspace(std::make_shared<Subspace>(space_.get(),lower,upper,space_name)); //register the full space as its trivial subspace under the same name
 assert(id == FULL_SUBSPACE);
}


SpaceRegister::SpaceRegister()
{
 static_assert(SOME_SPACE == 0);
 spaces_.emplace_back(SpaceRegEntry(std::make_shared<VectorSpace>(MAX_SPACE_DIM))); //Space 0 (SOME_SPACE) is an unnamed abstract space of maximal dimension
}

SpaceId SpaceRegister::registerSpace(std::shared_ptr<VectorSpace> & space)
{
 SpaceId id = space->getRegisteredId();
 if(id == SOME_SPACE){ //previously unregistered
  id = spaces_.size(); //new registered space id (>0)
  bool unique = name2id_.insert({space->getName(),id}).second;
  if(unique){
   space->resetRegisteredId(id);
   spaces_.emplace_back(SpaceRegEntry(space)); //space register shares ownership of the stored space
  }else{
   std::cout << "WARNING: Attempt to register vector space with an already registered name: " << space->getName() << std::endl;
   assert(unique); //space with this name already exists, registration unsuccessful
  }
 }
 return id;
}

SpaceId SpaceRegister::registerSpace(std::shared_ptr<VectorSpace> && space)
{
 SpaceId id = space->getRegisteredId();
 if(id == SOME_SPACE){ //previously unregistered
  id = spaces_.size(); //new registered space id (>0)
  bool unique = name2id_.insert({space->getName(),id}).second;
  if(unique){
   space->resetRegisteredId(id);
   spaces_.emplace_back(SpaceRegEntry(space)); //space register shares ownership of the stored space
  }else{
   std::cout << "WARNING: Attempt to register vector space with an already registered name: " << space->getName() << std::endl;
   assert(unique); //space with this name already exists, registration unsuccessful
  }
 }
 return id;
}


VectorSpace & SpaceRegister::getSpace(SpaceId id) const
{
 assert(id < spaces_.size());
 return *(spaces_[id].space_);
}

VectorSpace & SpaceRegister::getSpace(const std::string & name) const
{
 auto it = name2id_.find(name); assert(it != name2id_.end());
 return *(spaces_[it->second].space_);
}

} //namespace numerics

} //namespace exatn
