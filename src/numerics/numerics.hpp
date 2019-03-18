/** ExaTN::Numerics: Client Header
REVISION: 2019/03/17

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 1. Vector space and subspace registration:
    (a) Any unnamed vector space is automatically associated with a preregistered
        anonymous vector space wtih id = SOME_SPACE = 0.
    (b) Any explicitly registered (named) vector space has id > 0.
    (c) Any unregistered subspace of any vector space has id = UNREG_SUBSPACE = max(uint64_t).
    (d) Every explicitly registered (named) vector space has an automatically registered full
        subspace (=space) under the smae (space) name with id = FULL_SUBSPACE = 0.
    (e) Every registered non-trivial subspace of any vector space, including anonymous, has id:
        0 < id < max(uint64_t).
 2. Index labels:
    (a) Any registered subspace can be assigned a symbolic index label serving as a placeholder for it;
        any index label can only refer to a single registered (named) subspace it is associated with.
**/

#ifndef NUMERICS_HPP_
#define NUMERICS_HPP_

#include "numerics_factory.hpp"

#include <utility>
#include <memory>
#include <string>

namespace exatn{

using ScopeId = unsigned int; //TAProL scope ID type

namespace numerics{

/** Opens a new (child) TAProL scope and returns its id. **/
ScopeId openScope(const std::string & scope_name);

/** Closes the currently open TAProL scope and returns its parental scope id. **/
ScopeId closeScope();


/** Creates a named vector space, returns its registered id, and,
    optionally, a non-owning pointer to it. **/
SpaceId createVectorSpace(const std::string & space_name,            //in: vector space name
                          DimExtent space_dim,                       //in: vector space dimension
                          const VectorSpace ** space_ptr = nullptr); //out: non-owning pointer to the created vector space

/** Destroys a previously created vector space. **/
void destroyVectorSpace(const std::string & space_name); //in: name of the vector space to destroy
void destroyVectorSpace(SpaceId space_id);               //in: id of the vector space to destroy

/** Creates a named subspace of a named vector space,
    returns its registered id, and, optionally, a non-owning pointer to it. **/
SubspaceId createSubspace(const std::string & subspace_name,           //in: subspace name
                          const std::string & space_name,              //in: containing vector space name
                          const std::pair<DimOffset,DimOffset> bounds, //in: range of basis vectors defining the subspace: [lower:upper]
                          const Subspace ** subspace_ptr = nullptr);   //out: non-owning pointer to the created subspace

/** Destroys a previously created subspace. **/
void destroySubspace(const std::string & subspace_name); //in: name of the subspace to destroy
void destroySubspace(SubspaceId subspace_id);            //in: id of the subspace to destroy

} //namespace numerics

} //namespace exatn

#endif //NUMERICS_HPP_
