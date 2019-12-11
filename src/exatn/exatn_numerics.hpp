/** ExaTN::Numerics: General client header
REVISION: 2019/12/11

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
inline ScopeId openScope(const std::string & scope_name) //new scope name
 {return numericalServer->openScope(scope_name);}


/** Closes the currently open TAProL scope and returns its parental scope id. **/
inline ScopeId closeScope()
 {return numericalServer->closeScope();}


/** Creates a named vector space, returns its registered id, and,
    optionally, a non-owning pointer to it. **/
inline SpaceId createVectorSpace(const std::string & space_name,           //in: vector space name
                                 DimExtent space_dim,                      //in: vector space dimension
                                 const VectorSpace ** space_ptr = nullptr) //out: non-owning pointer to the created vector space
 {return numericalServer->createVectorSpace(space_name,space_dim,space_ptr);}


/** Destroys a previously created named vector space. **/
inline void destroyVectorSpace(const std::string & space_name) //in: name of the vector space to destroy
 {return numericalServer->destroyVectorSpace(space_name);}

inline void destroyVectorSpace(SpaceId space_id) //in: id of the vector space to destroy
 {return numericalServer->destroyVectorSpace(space_id);}


/** Creates a named subspace of a named vector space,
    returns its registered id, and, optionally, a non-owning pointer to it. **/
inline SubspaceId createSubspace(const std::string & subspace_name,           //in: subspace name
                                 const std::string & space_name,              //in: containing vector space name
                                 const std::pair<DimOffset,DimOffset> bounds, //in: range of basis vectors defining the created subspace: [lower:upper]
                                 const Subspace ** subspace_ptr = nullptr)    //out: non-owning pointer to the created subspace
 {return numericalServer->createSubspace(subspace_name,space_name,bounds,subspace_ptr);}


/** Destroys a previously created named subspace of a named vector space. **/
inline void destroySubspace(const std::string & subspace_name) //in: name of the subspace to destroy
 {return numericalServer->destroySubspace(subspace_name);}

inline void destroySubspace(SubspaceId subspace_id) //in: id of the subspace to destroy
 {return numericalServer->destroySubspace(subspace_id);}


/** Returns a non-owning pointer to a previosuly registered named subspace
    of a previously registered named vector space. **/
inline const Subspace * getSubspace(const std::string & subspace_name) //in: name of the subspace to get
 {return numericalServer->getSubspace(subspace_name);}


/** Registers an external tensor method. **/
inline void registerTensorMethod(const std::string & tag,
                                 std::shared_ptr<TensorMethod> method)
 {return numericalServer->registerTensorMethod(tag,method);}


/** Retrieves a registered external tensor method. **/
inline std::shared_ptr<TensorMethod> getTensorMethod(const std::string & tag)
 {return numericalServer->getTensorMethod(tag);}


/** Registers an external data packet. **/
inline void registerExternalData(const std::string & tag,
                                 std::shared_ptr<BytePacket> packet)
 {return numericalServer->registerExternalData(tag,packet);}


/** Retrieves a registered external data packet. **/
inline std::shared_ptr<BytePacket> getExternalData(const std::string & tag)
 {return numericalServer->getExternalData(tag);}


/** Declares, registers and actually creates a tensor via processing backend.
    See numerics::Tensor constructors for different creation options. **/
template <typename... Args>
inline bool createTensor(const std::string & name,       //in: tensor name
                         TensorElementType element_type, //in: tensor element type
                         Args&&... args)                 //in: other arguments for Tensor ctor
 {return numericalServer->createTensor(name,element_type,std::forward<Args>(args)...);}

inline bool createTensor(std::shared_ptr<Tensor> tensor, //in: existing declared tensor
                         TensorElementType element_type) //in: tensor element type
 {return numericalServer->createTensor(tensor,element_type);}

template <typename... Args>
inline bool createTensorSync(const std::string & name,       //in: tensor name
                             TensorElementType element_type, //in: tensor element type
                             Args&&... args)                 //in: other arguments for Tensor ctor
 {return numericalServer->createTensorSync(name,element_type,std::forward<Args>(args)...);}

inline bool createTensorSync(std::shared_ptr<Tensor> tensor, //in: existing declared tensor
                             TensorElementType element_type) //in: tensor element type
 {return numericalServer->createTensorSync(tensor,element_type);}


/** Returns a shared pointer to the actual tensor object. **/
inline std::shared_ptr<Tensor> getTensor(const std::string & name) //in: tensor name
 {return numericalServer->getTensor(name);}


/** Returns the reference to the actual tensor object. **/
inline Tensor & getTensorRef(const std::string & name) //in: tensor name
 {return numericalServer->getTensorRef(name);}


/** Returns the tensor element type. **/
inline TensorElementType getTensorElementType(const std::string & name) //in: tensor name
 {return numericalServer->getTensorElementType(name);}


/** Registers a group of tensor dimensions which form an isometry when
    contracted over with the conjugated tensor (see exatn::numerics::Tensor).
    Returns TRUE on success, FALSE on failure. **/
inline bool registerTensorIsometry(const std::string & name,                   //in: tensor name
                                   const std::vector<unsigned int> & iso_dims) //in: tensor dimensions forming the isometry
 {return numericalServer->registerTensorIsometry(name,iso_dims);}

inline bool registerTensorIsometry(const std::string & name,                    //in: tensor name
                                   const std::vector<unsigned int> & iso_dims0, //in: tensor dimensions forming the isometry (group 0)
                                   const std::vector<unsigned int> & iso_dims1) //in: tensor dimensions forming the isometry (group 1)
 {return numericalServer->registerTensorIsometry(name,iso_dims0,iso_dims1);}


/** Destroys a tensor, including its backend representation. **/
inline bool destroyTensor(const std::string & name) //in: tensor name
 {return numericalServer->destroyTensor(name);}

inline bool destroyTensorSync(const std::string & name) //in: tensor name
 {return numericalServer->destroyTensorSync(name);}


/** Initializes a tensor to some scalar value. **/
template<typename NumericType>
inline bool initTensor(const std::string & name, //in: tensor name
                       NumericType value)        //in: scalar value
 {return numericalServer->initTensor(name,value);}

template<typename NumericType>
inline bool initTensorSync(const std::string & name, //in: tensor name
                           NumericType value)        //in: scalar value
 {return numericalServer->initTensorSync(name,value);}


/** Initializes a tensor with externally provided data. **/
template<typename NumericType>
inline bool initTensorData(const std::string & name,                  //in: tensor name
                           const std::vector<NumericType> & ext_data) //in: externally provided data
 {return numericalServer->initTensorData(name,ext_data);}

template<typename NumericType>
inline bool initTensorDataSync(const std::string & name,                  //in: tensor name
                               const std::vector<NumericType> & ext_data) //in: externally provided data
 {return numericalServer->initTensorDataSync(name,ext_data);}


/** Initializes the tensor body with random values. **/
inline bool initTensorRnd(const std::string & name) //in: tensor name
 {return numericalServer->initTensorRnd(name);}

inline bool initTensorRndSync(const std::string & name) //in: tensor name
 {return numericalServer->initTensorRndSync(name);}


/** Transforms (updates) a tensor according to a user-defined tensor functor. **/
inline bool transformTensor(const std::string & name,              //in: tensor name
                            std::shared_ptr<TensorMethod> functor) //in: functor defining tensor transformation
 {return numericalServer->transformTensor(name,functor);}

inline bool transformTensorSync(const std::string & name,              //in: tensor name
                                std::shared_ptr<TensorMethod> functor) //in: functor defining tensor transformation
 {return numericalServer->transformTensorSync(name,functor);}


/** Performs tensor addition: tensor0 += tensor1 * alpha **/
template<typename NumericType>
inline bool addTensors(const std::string & addition, //in: symbolic tensor addition specification
                       NumericType alpha)            //in: alpha prefactor
 {return numericalServer->addTensors(addition,alpha);}

template<typename NumericType>
inline bool addTensorsSync(const std::string & addition, //in: symbolic tensor addition specification
                           NumericType alpha)            //in: alpha prefactor
 {return numericalServer->addTensorsSync(addition,alpha);}


/** Performs tensor contraction: tensor0 += tensor1 * tensor2 * alpha **/
template<typename NumericType>
inline bool contractTensors(const std::string & contraction, //in: symbolic tensor contraction specification
                            NumericType alpha)               //in: alpha prefactor
 {return numericalServer->contractTensors(contraction,alpha);}

template<typename NumericType>
inline bool contractTensorsSync(const std::string & contraction, //in: symbolic tensor contraction specification
                                NumericType alpha)               //in: alpha prefactor
 {return numericalServer->contractTensorsSync(contraction,alpha);}


/** Performs a full evaluation of a tensor network specified symbolically, based on
    the symbolic names of previously created tensors (including the output tensor). **/
inline bool evaluateTensorNetwork(const std::string & name,    //in: tensor network name
                                  const std::string & network) //in: symbolic tensor network specification
 {return numericalServer->evaluateTensorNetwork(name,network);}

inline bool evaluateTensorNetworkSync(const std::string & name,    //in: tensor network name
                                      const std::string & network) //in: symbolic tensor network specification
 {return numericalServer->evaluateTensorNetworkSync(name,network);}


/** Synchronizes all outstanding update operations on a given tensor
    specified by its symbolic name. **/
inline bool sync(const std::string & name, //in: tensor name
                 bool wait = true)         //in: wait versus test for completion
 {return numericalServer->sync(name,wait);}


/** Evaluates a tensor network object (computes the output tensor). **/
inline bool evaluate(TensorNetwork & network) //in: finalized tensor network
 {return numericalServer->submit(network);}

inline bool evaluateSync(TensorNetwork & network) //in: finalized tensor network
 {bool success = numericalServer->submit(network);
  if(success) success = numericalServer->sync(network);
  return success;}


/** Synchronizes all outstanding operations on a given tensor network object. **/
inline bool sync(TensorNetwork & network, //in: finalized tensor network
                 bool wait = true)        //in: wait versus test for completion
 {return numericalServer->sync(network,wait);}


/** Evaluates a tensor network expansion into the explicitly provided tensor accumulator. **/
inline bool evaluate(TensorExpansion & expansion,         //in: tensor network expansion
                     std::shared_ptr<Tensor> accumulator) //inout: tensor accumulator
 {return numericalServer->submit(expansion,accumulator);}

inline bool evaluateSync(TensorExpansion & expansion,         //in: tensor network expansion
                         std::shared_ptr<Tensor> accumulator) //inout: tensor accumulator
 {if(!accumulator) return false;
  bool success = numericalServer->submit(expansion,accumulator);
  if(success) success = numericalServer->sync(*accumulator);
  return success;}


/** Synchronizes all outstanding operations on a given tensor. **/
inline bool sync(const Tensor & tensor, //in: tensor
                 bool wait = true)      //in: wait versus test for completion
 {return numericalServer->sync(tensor,wait);}


/** Synchronizes all outstanding tensor operations in the current scope (barrier). **/
inline bool sync(bool wait = true)
 {return numericalServer->sync(wait);}


/** Returns a locally stored tensor slice (talsh::Tensor) providing access to tensor elements.
    This slice will be extracted from the exatn::numerics::Tensor implementation as a copy.
    The returned future becomes ready once the execution thread has retrieved the slice copy. **/
inline std::shared_ptr<talsh::Tensor> getLocalTensor(std::shared_ptr<Tensor> tensor, //in: exatn::numerics::Tensor to get slice of (by copy)
                     const std::vector<std::pair<DimOffset,DimExtent>> & slice_spec) //in: tensor slice specification
 {return numericalServer->getLocalTensor(tensor,slice_spec);}

inline std::shared_ptr<talsh::Tensor> getLocalTensor(const std::string & name, //in: name of the registered exatn::numerics::Tensor
               const std::vector<std::pair<DimOffset,DimExtent>> & slice_spec) //in: tensor slice specification
 {return numericalServer->getLocalTensor(name,slice_spec);}

inline std::shared_ptr<talsh::Tensor> getLocalTensor(const std::string & name) //in: name of the registered exatn::numerics::Tensor
 {return numericalServer->getLocalTensor(name);}


/** Resets tensor runtime logging level (0:none). **/
inline void resetRuntimeLoggingLevel(int level = 0)
 {return numericalServer->resetRuntimeLoggingLevel(level);}

} //namespace exatn

#endif //EXATN_NUMERICS_HPP_
