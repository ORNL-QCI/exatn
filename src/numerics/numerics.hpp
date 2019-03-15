/** ExaTN::Numerics: Client Header
REVISION: 2019/03/15

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/* Rules:
 1. Vector space and subspace registration:
    (a) Any unregistered vector space has id = SOME_SPACE = 0.
    (b) Any registered vector space has id > 0.
    (c) Any unregistered subspace of any vector space has id = UNREG_SUBSPACE = max(uint64_t).
    (d) Every registered vector space has an automatically registered full subspace under the space name with id 0 (FULL_SUBSPACE).
    (e) Every registered non-trivial subspace of any registered vector space has id: 0 < id < max(uint64_t).
 2. Index labels:
    (a) Any subspace can be assigned a symbolic index label; the index label cannot refer to multiple subspaces.
*/

#ifndef NUMERICS_HPP_
#define NUMERICS_HPP_

#include "register.hpp"

#include <memory>
#include <string>

namespace exatn{

namespace numerics{

//PUBLIC:

/** Opens a new scope and returns its id. **/
unsigned int openScope(const std::string & scope_name);

/** Closes the currently open scope. **/
void closeScope();

/** Registers a vector space and returns its unique registered id.
    Space register shares ownership of the registered space. **/
SpaceId registerVectorSpace(std::shared_ptr<VectorSpace> & space);

/** Registers a non-trivial non-empty subspace of a vector space and returns its unique
    registered id. Subspace register shares ownership of the registered subspace. **/
SubspaceId registerVectorSubspace(std::shared_ptr<Subspace> & subspace);

/** Registers a symbolic index label to refer to a particular subspace and returns
    its unique registered id. The subspace itself does not have to be registered. **/
unsigned long registerIndexLabel(const std::string & index_label, const Subspace & subspace);

/** Unregisters a previously registered index label. **/
bool unregisterIndexLabel(const std::string & index_label);


//PRIVATE:

/** Space/subspace register **/
SpaceRegister space_register;

} //namespace numerics

} //namespace exatn

#endif //NUMERICS_HPP_
