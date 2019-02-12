/** ExaTN::Numerics: Register
REVISION: 2019/02/12

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef REGISTER_HPP_
#define REGISTER_HPP_

#include "tensor_basic.hpp"
#include "spaces.hpp"

#include <memory>
#include <string>
#include <vector>

namespace exatn{

namespace numerics{

class SubspaceRegEntry{
public:

 SubspaceRegEntry(std::shared_ptr<Subspace> & subspace);
 SubspaceRegEntry(std::shared_ptr<Subspace> && subspace);

 SubspaceRegEntry(const SubspaceRegEntry &) = delete;
 SubspaceRegEntry & operator=(const SubspaceRegEntry &) = delete;
 SubspaceRegEntry(SubspaceRegEntry &&) = default;
 SubspaceRegEntry & operator=(SubspaceRegEntry &&) = default;
 virtual ~SubspaceRegEntry() = default;

private:

 std::shared_ptr<Subspace> subspace_; //registered subspace of a vector space
};


class SubspaceRegister{
public:

 SubspaceRegister() = default;

 SubspaceRegister(const SubspaceRegister &) = delete;
 SubspaceRegister & operator=(const SubspaceRegister &) = delete;
 SubspaceRegister(SubspaceRegister &&) = default;
 SubspaceRegister & operator=(SubspaceRegister &&) = default;
 virtual ~SubspaceRegister() = default;

 /** Registers a subspace of some vector space and returns its registered id. **/
 SubspaceId registerSubspace(std::shared_ptr<Subspace> & subspace);
 SubspaceId registerSubspace(std::shared_ptr<Subspace> && subspace);

private:

 std::vector<SubspaceRegEntry> subspaces_; //registered subspaces of some vector space
};


class SpaceRegEntry{
public:

 SpaceRegEntry(std::shared_ptr<VectorSpace> & space);
 SpaceRegEntry(std::shared_ptr<VectorSpace> && space);

 SpaceRegEntry(const SpaceRegEntry &) = delete;
 SpaceRegEntry & operator=(const SpaceRegEntry &) = delete;
 SpaceRegEntry(SpaceRegEntry &&) = default;
 SpaceRegEntry & operator=(SpaceRegEntry &&) = default;
 virtual ~SpaceRegEntry() = default;

private:

 std::shared_ptr<VectorSpace> space_; //registered vector space
 SubspaceRegister subspaces_;         //subspace register for this vector space
};


class SpaceRegister{
public:

 SpaceRegister();

 SpaceRegister(const SpaceRegister &) = delete;
 SpaceRegister & operator=(const SpaceRegister &) = delete;
 SpaceRegister(SpaceRegister &&) = default;
 SpaceRegister & operator=(SpaceRegister &&) = default;
 virtual ~SpaceRegister() = default;

 /** Registers a vector space and returns its registered id. **/
 SpaceId registerSpace(std::shared_ptr<VectorSpace> & space);
 SpaceId registerSpace(std::shared_ptr<VectorSpace> && space);

private:

 std::vector<SpaceRegEntry> spaces_; //registered vector spaces
};

} //namespace numerics

} //namespace exatn

#endif //REGISTER_HPP_
