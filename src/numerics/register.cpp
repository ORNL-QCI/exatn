/** ExaTN::Numerics: Register
REVISION: 2019/04/22

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
  const std::string & subspace_name = subspace->getName();
  assert(subspace_name.length() > 0); //only named subspaces can be registered
  id = subspaces_.size(); //new registered subspace id (>0)
  bool unique = name2id_.insert({subspace_name,id}).second;
  if(unique){
   subspace->resetRegisteredId(id);
   subspaces_.emplace_back(SubspaceRegEntry(subspace)); //subspace register shares ownership of the stored subspace
  }else{
   std::cout << "WARNING: Attempt to register a subspace with an already registered name: " << subspace_name << std::endl;
   return UNREG_SUBSPACE; //subspace with this name already exists, subspace cannot be registered
  }
 }
 return id;
}

SubspaceId SubspaceRegister::registerSubspace(std::shared_ptr<Subspace> && subspace)
{
 SubspaceId id = subspace->getRegisteredId();
 if(id == UNREG_SUBSPACE){ //previously unregistered
  const std::string & subspace_name = subspace->getName();
  assert(subspace_name.length() > 0); //only named subspaces can be registered
  id = subspaces_.size(); //new registered subspace id (>0)
  bool unique = name2id_.insert({subspace_name,id}).second;
  if(unique){
   subspace->resetRegisteredId(id);
   subspaces_.emplace_back(SubspaceRegEntry(subspace)); //subspace register shares ownership of the stored subspace
  }else{
   std::cout << "WARNING: Attempt to register a subspace with an already registered name: " << subspace_name << std::endl;
   return UNREG_SUBSPACE; //subspace with this name already exists, subspace cannot be registered
  }
 }
 return id;
}


const Subspace * SubspaceRegister::getSubspace(SubspaceId id) const
{
 if(id >= subspaces_.size()) return nullptr;
 return subspaces_[id].subspace_.get();
}

const Subspace * SubspaceRegister::getSubspace(const std::string & name) const
{
 auto it = name2id_.find(name);
 if(it == name2id_.end()) return nullptr;
 return subspaces_[it->second].subspace_.get();
}


SpaceRegEntry::SpaceRegEntry(std::shared_ptr<VectorSpace> & space):
 space_(space)
{
 DimOffset lower = 0;
 DimOffset upper = lower + space_->getDimension() - 1;
 const std::string & space_name = space_->getName();
 if(space_name.length() > 0){
  SubspaceId id = subspaces_.registerSubspace(std::make_shared<Subspace>(space_.get(),lower,upper,space_name)); //register the full space as its trivial subspace under the same name
  assert(id == FULL_SUBSPACE); //=0
 }
}

SpaceRegEntry::SpaceRegEntry(std::shared_ptr<VectorSpace> && space):
 space_(space)
{
 DimOffset lower = 0;
 DimOffset upper = lower + space_->getDimension() - 1;
 const std::string & space_name = space_->getName();
 if(space_name.length() > 0){
  SubspaceId id = subspaces_.registerSubspace(std::make_shared<Subspace>(space_.get(),lower,upper,space_name)); //register the full space as its trivial subspace under the same name
  assert(id == FULL_SUBSPACE); //=0
 }
}


SpaceRegister::SpaceRegister()
{
 static_assert(SOME_SPACE == 0, "#FATAL(exatn::numerics::SpaceRegister): Predefined SOME_SPACE space id is not equal 0!");
 spaces_.emplace_back(SpaceRegEntry(std::make_shared<VectorSpace>(MAX_SPACE_DIM))); //Space 0 (SOME_SPACE) is an unnamed abstract space of maximal dimension
}

SpaceId SpaceRegister::registerSpace(std::shared_ptr<VectorSpace> & space)
{
 SpaceId id = space->getRegisteredId();
 if(id == SOME_SPACE){ //previously unregistered
  const std::string & space_name = space->getName();
  assert(space_name.length() > 0); //only named spaces can be registered explicitly
  id = spaces_.size(); //new registered space id (>0)
  bool unique = name2id_.insert({space_name,id}).second;
  if(unique){
   space->resetRegisteredId(id);
   spaces_.emplace_back(SpaceRegEntry(space)); //space register shares ownership of the stored vector space
  }else{
   std::cout << "WARNING: Attempt to register a vector space with an already registered name: " << space_name << std::endl;
   return SOME_SPACE; //space with this name already exists, space cannot be registered
  }
 }
 return id;
}

SpaceId SpaceRegister::registerSpace(std::shared_ptr<VectorSpace> && space)
{
 SpaceId id = space->getRegisteredId();
 if(id == SOME_SPACE){ //previously unregistered
  const std::string & space_name = space->getName();
  assert(space_name.length() > 0); //only named spaces can be registered explicitly
  id = spaces_.size(); //new registered space id (>0)
  bool unique = name2id_.insert({space_name,id}).second;
  if(unique){
   space->resetRegisteredId(id);
   spaces_.emplace_back(SpaceRegEntry(space)); //space register shares ownership of the stored vector space
  }else{
   std::cout << "WARNING: Attempt to register a vector space with an already registered name: " << space_name << std::endl;
   return SOME_SPACE; //space with this name already exists, space cannot be registered
  }
 }
 return id;
}


const VectorSpace * SpaceRegister::getSpace(SpaceId id) const
{
 if(id >= spaces_.size()) return nullptr;
 return spaces_[id].space_.get();
}

const VectorSpace * SpaceRegister::getSpace(const std::string & name) const
{
 if(name.length() > 0){ //named vector space
  auto it = name2id_.find(name);
  if(it == name2id_.end()) return nullptr;
  return spaces_[it->second].space_.get();
 }else{ //unnamed vector space (space 0 = SOME_SPACE)
  return spaces_[0].space_.get();
 }
}

} //namespace numerics

} //namespace exatn
