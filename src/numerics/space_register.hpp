/** ExaTN::Numerics: Register of vector spaces and their subspaces
REVISION: 2020/02/28

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Any unregistered vector space has id = SOME_SPACE = 0 (anonymous vector space).
     A subspace of the anonymous vector space is defined by
     the base offset (first basis vector) and its dimension.
 (b) Any explicitly registered (named) vector space has id > 0.
 (c) Any unregistered subspace of any registered vector space has id = UNREG_SUBSPACE = max(uint64_t).
 (d) Every named vector space has an automatically registered full subspace under
     the same (space) name with id = FULL_SUBSPACE = 0 (trivial subspace which spans the full space).
 (e) Every registered non-trivial subspace of any registered vector space has id: 0 < id < max(uint64_t).
**/

#ifndef EXATN_NUMERICS_SPACE_REGISTER_HPP_
#define EXATN_NUMERICS_SPACE_REGISTER_HPP_

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

 SubspaceRegEntry(std::shared_ptr<Subspace> subspace);

 SubspaceRegEntry(const SubspaceRegEntry &) = default;
 SubspaceRegEntry & operator=(const SubspaceRegEntry &) = default;
 SubspaceRegEntry(SubspaceRegEntry &&) noexcept = default;
 SubspaceRegEntry & operator=(SubspaceRegEntry &&) noexcept = default;
 ~SubspaceRegEntry() = default;

 friend class SubspaceRegister;

private:

 std::shared_ptr<Subspace> subspace_; //registered subspace of a vector space (owned by RegEntry)
};


class SubspaceRegister{
public:

 SubspaceRegister() = default;

 SubspaceRegister(const SubspaceRegister &) = delete;
 SubspaceRegister & operator=(const SubspaceRegister &) = delete;
 SubspaceRegister(SubspaceRegister &&) noexcept = default;
 SubspaceRegister & operator=(SubspaceRegister &&) noexcept = default;
 ~SubspaceRegister() = default;

 /** Registers a subspace of some vector space and returns its registered id.
 If the subspace has already been registered before, returns its existing id.
 Returned id = UNREG_SUBSPACE means that another subspace with the same name
 has already been registered before. **/
 SubspaceId registerSubspace(std::shared_ptr<Subspace> subspace);

 /** Returns a non-owning pointer to a stored subspace by its id. **/
 const Subspace * getSubspace(SubspaceId id) const;
 /** Returns a non-owning pointer to a stored subspace by its symbolic name. **/
 const Subspace * getSubspace(const std::string & name) const;

private:

 std::vector<SubspaceRegEntry> subspaces_; //registered subspaces of some vector space
 std::unordered_map<std::string,SubspaceId> name2id_; //maps subspace names to their integer ids
};


class SpaceRegEntry{
public:

 SpaceRegEntry(std::shared_ptr<VectorSpace> space);

 SpaceRegEntry(const SpaceRegEntry &) = delete;
 SpaceRegEntry & operator=(const SpaceRegEntry &) = delete;
 SpaceRegEntry(SpaceRegEntry &&) noexcept = default;
 SpaceRegEntry & operator=(SpaceRegEntry &&) noexcept = default;
 ~SpaceRegEntry() = default;

 friend class SpaceRegister;

private:

 std::shared_ptr<VectorSpace> space_; //registered vector space (owned by RegEntry)
 SubspaceRegister subspaces_;         //subspace register for this vector space
};


class SpaceRegister{
public:

 SpaceRegister();

 SpaceRegister(const SpaceRegister &) = delete;
 SpaceRegister & operator=(const SpaceRegister &) = delete;
 SpaceRegister(SpaceRegister &&) noexcept = default;
 SpaceRegister & operator=(SpaceRegister &&) noexcept = default;
 ~SpaceRegister() = default;

 /** Registers a named vector space and returns its registered id.
 If it has already been registered before, returns its existing id.
 Returned id = SOME_SPACE indicates an attempt to register a named
 vector space when another vector space has already been registered
 under the same name. **/
 SpaceId registerSpace(std::shared_ptr<VectorSpace> space);

 /** Returns a non-owning pointer to a stored vector space by its id. **/
 const VectorSpace * getSpace(SpaceId id) const;
 /** Returns a non-owning pointer to a stored vector space by its symbolic name. **/
 const VectorSpace * getSpace(const std::string & name) const;

 /** Registers a named subspace of a named vector space and returns its registered id.
 If the subspace has already been registered before, returns its existing id.
 Returned id = UNREG_SUBSPACE means that another subspace with the same name
 has already been registered before. **/
 SubspaceId registerSubspace(std::shared_ptr<Subspace> subspace);

 /** Returns a non-owning pointer to a registerd subspace of a registered vector space. **/
 const Subspace * getSubspace(const std::string & space_name,
                              const std::string & subspace_name) const;

private:

 std::vector<SpaceRegEntry> spaces_; //registered vector spaces
 std::unordered_map<std::string,SpaceId> name2id_; //maps vector space names to their integer ids
};

} //namespace numerics

/** Returns the global register of vector spaces and subspaces. **/
std::shared_ptr<numerics::SpaceRegister> getSpaceRegister();

} //namespace exatn

#endif //EXATN_NUMERICS_SPACE_REGISTER_HPP_
