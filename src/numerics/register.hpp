/** ExaTN::Numerics: Register
REVISION: 2019/03/15

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/* NOTES:
 (a) Any unregistered vector space has id = SOME_SPACE = 0.
 (b) Any registered vector space has id > 0.
 (c) Any unregistered subspace of any vector space has id = UNREG_SUBSPACE = max(uint64_t).
 (d) Every registered vector space has an automatically registered full subspace under the space name with id 0.
 (e) Every registered non-trivial subspace of any registered vector space has id: 0 < id < max(uint64_t).
*/

#ifndef REGISTER_HPP_
#define REGISTER_HPP_

#include "tensor_basic.hpp"
#include "spaces.hpp"

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

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

 friend class SubspaceRegister;

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

 /** Returns a stored subspace by its id. **/
 Subspace & getSubspace(SubspaceId id) const;
 /** Returns a stored subspace by its symbolic name. **/
 Subspace & getSubspace(const std::string & name) const;

private:

 std::vector<SubspaceRegEntry> subspaces_; //registered subspaces of some vector space
 std::unordered_map<std::string,SubspaceId> name2id_; //maps subspace names to their integer ids
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

 friend class SpaceRegister;

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

 /** Returns a stored space by its id. **/
 VectorSpace & getSpace(SpaceId id) const;
 /** Returns a stored subspace by its symbolic name. **/
 VectorSpace & getSpace(const std::string & name) const;

private:

 std::vector<SpaceRegEntry> spaces_; //registered vector spaces
 std::unordered_map<std::string,SpaceId> name2id_; //maps space names to their integer ids
};

} //namespace numerics

} //namespace exatn

#endif //REGISTER_HPP_
