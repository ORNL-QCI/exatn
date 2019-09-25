/** ExaTN::Numerics: General client header
REVISION: 2019/09/25

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 1. Vector space and subspace registration:
    (a) Any unnamed vector space is automatically associated with a preregistered
        anonymous vector space wtih id = SOME_SPACE = 0.
    (b) Any explicitly registered (named) vector space has id > 0.
    (c) Any unregistered subspace of any named vector space has id = UNREG_SUBSPACE = max(uint64_t).
    (d) Every explicitly registered (named) vector space has an automatically registered full
        subspace (=space) under the same (space) name with id = FULL_SUBSPACE = 0.
    (e) Every registered non-trivial named subspace of any named vector space has id:
        0 < id < max(uint64_t).
    (f) A subspace of the anonymous vector space is defined by the
        base offset (first basis vector belonging to it) and its dimension.
 2. Index labels:
    (a) Any registered subspace can be assigned a symbolic index label serving as a placeholder for it;
        any index label can only refer to a single registered (named) subspace it is associated with.
**/

#ifndef EXATN_NUMERICS_HPP_
#define EXATN_NUMERICS_HPP_

#include "num_server.hpp"

#include <utility>
#include <memory>
#include <string>

namespace exatn{

/** Opens a new (child) TAProL scope and returns its id. **/
ScopeId openScope(const std::string & scope_name) //new scope name
 {return numericalServer->openScope(scope_name);}

/** Closes the currently open TAProL scope and returns its parental scope id. **/
ScopeId closeScope()
 {return numericalServer->closeScope();}


/** Creates a named vector space, returns its registered id, and,
    optionally, a non-owning pointer to it. **/
SpaceId createVectorSpace(const std::string & space_name,           //in: vector space name
                          DimExtent space_dim,                      //in: vector space dimension
                          const VectorSpace ** space_ptr = nullptr) //out: non-owning pointer to the created vector space
 {return numericalServer->createVectorSpace(space_name,space_dim,space_ptr);}

/** Destroys a previously created named vector space. **/
void destroyVectorSpace(const std::string & space_name) //in: name of the vector space to destroy
 {return numericalServer->destroyVectorSpace(space_name);}

void destroyVectorSpace(SpaceId space_id) //in: id of the vector space to destroy
 {return numericalServer->destroyVectorSpace(space_id);}

/** Creates a named subspace of a named vector space,
    returns its registered id, and, optionally, a non-owning pointer to it. **/
SubspaceId createSubspace(const std::string & subspace_name,           //in: subspace name
                          const std::string & space_name,              //in: containing vector space name
                          const std::pair<DimOffset,DimOffset> bounds, //in: range of basis vectors defining the created subspace: [lower:upper]
                          const Subspace ** subspace_ptr = nullptr)    //out: non-owning pointer to the created subspace
 {return numericalServer->createSubspace(subspace_name,space_name,bounds,subspace_ptr);}

/** Destroys a previously created named subspace of a named vector space. **/
void destroySubspace(const std::string & subspace_name) //in: name of the subspace to destroy
 {return numericalServer->destroySubspace(subspace_name);}

void destroySubspace(SubspaceId subspace_id) //in: id of the subspace to destroy
 {return numericalServer->destroySubspace(subspace_id);}

/** Returns a non-owning pointer to a previosuly registered named subspace
    of a previously registered named vector space. **/
const Subspace * getSubspace(const std::string & subspace_name) //in: name of the subspace to get
 {return numericalServer->getSubspace(subspace_name);}

/** Declares, registers and actually creates a tensor via processing backend.
    See numerics::Tensor constructors for different creation options. **/
template <typename... Args>
bool createTensor(const std::string & name,       //in: tensor name
                  TensorElementType element_type, //in: tensor element type
                  Args&&... args)                 //in: other arguments for Tensor ctor
 {return numericalServer->createTensor(name,element_type,args...);}

/** Returns the reference to the actual tensor object. **/
Tensor & getTensorRef(const std::string & name) //in: tensor name
 {return numericalServer->getTensorRef(name);}

/** Returns the tensor element type. **/
TensorElementType getTensorElementType(const std::string & name) //in: tensor name
 {return numericalServer->getTensorElementType(name);}

/** Destroys a tensor, including its backend representation. **/
bool destroyTensor(const std::string & name) //in: tensor name
 {return numericalServer->destroyTensor(name);}

/** Initializes a tensor to some scalar value. **/
template<typename NumericType>
bool initTensor(const std::string & name, //in: tensor name
                NumericType value)        //in: scalar value
 {return numericalServer->initTensor(name,value);}

/** Transforms (updates) a tensor according to a user-defined tensor functor. **/
bool transformTensor(const std::string & name,              //in: tensor name
                     std::shared_ptr<TensorMethod> functor) //in: functor defining tensor transformation
 {return numericalServer->transformTensor(name,functor);}

/** Performs tensor addition: tensor0 += tensor1 * alpha **/
template<typename NumericType>
bool addTensors(const std::string & name0, //in: tensor 0 name
                const std::string & name1, //in: tensor 1 name
                NumericType alpha)         //in: alpha prefactor
 {return numericalServer->addTensors(name0,name1,alpha);}

/** Performs tensor contraction: tensor0 += tensor1 * tensor2 * alpha **/
template<typename NumericType>
bool contractTensors(const std::string & name0, //in: tensor 0 name
                     const std::string & name1, //in: tensor 1 name
                     const std::string & name2, //in: tensor 2 name
                     NumericType alpha)         //in: alpha prefactor
 {return numericalServer->contractTensors(name0,name1,name2,alpha);}

/** Performs a full evaluation of a tensor network. **/
bool evaluateTensorNetwork(const std::string & name,    //in: tensor network name
                           const std::string & network) //in: tensor network
 {return numericalServer->evaluateTensorNetwork(name,network);}

/** Synchronizes all outstanding update operations on a given tensor. **/
bool sync(const std::string & name, //in: tensor name
          bool wait = true)         //in: wait versus test for completion
 {return numericalServer->sync(name,wait);}

} //namespace exatn

#endif //EXATN_NUMERICS_HPP_
