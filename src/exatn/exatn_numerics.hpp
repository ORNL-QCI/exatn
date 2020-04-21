/** ExaTN::Numerics: General client header
REVISION: 2020/04/20

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 1. Vector space and subspace registration:
    (a) Any unnamed vector space is automatically associated with a pre-registered
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
 3. Tensor:
    (a) A tensor is defined by its name, shape and signature.
    (b) Tensor shape is an ordered list of tensor dimension extents.
    (c) Tensor signature is an ordered list of {space_id,subspace_id} pairs
        for each tensor dimension. In case space_id = SOME_SPACE, subspace_id
        is simply the base offset in the anonymous vector space (min = 0).
 4. Tensor operation:
    (a) Tensor operation is a mathematical operation on one or more tensor arguments.
    (b) Evaluating a tensor operation means computing the value of all its output tensors,
        given all input tensors.
 5. Tensor network:
    (a) Tensor network is a graph of tensors in which vertices are the tensors
        and (directed) edges show which dimensions of these tensors are contracted
        with each other. By default, each edge connects two dimensions in two separate
        tensors (vertices), although these tensors themselves may be identical.
    (b) The same tensor may enter the given tensor network multiple times
        via different vertices, either normal or conjugated.
    (c) Each tensor network has an implicit output tensor collecting all open
        edges from all input tensors. Evaluating the tensor network means
        computing the value of this output tensor, given all input tensors.
    (d) The conjugation operation applied to a tensor network performs complex conjugation
        of all constituent input tensors, but does not apply to the output tensor per se
        because the output tensor is simply the result of the full contraction of the tensor
        network. The conjugation operation also reverses the direction of all edges.
    (e) A single input tensor may enter multiple tensor networks.
 6. Tensor network expansion:
    (a) Tensor network expansion is a linear combination of tensor networks
        with some complex coefficients. The output tensors of all constituent
        tensor networks must be congruent. Evaluating the tensor network
        expansion means computing the sum of all these output tensors scaled
        by their respective coefficients.
    (b) A tensor network expansion may either belong to the ket or bra dual space.
        The conjugation operation transitions the tensor network expansion between
        the ket and bra spaces.
    (c) A single tensor network may enter multiple tensor network expansions.
 7. Tensor network operator:
    (a) Tensor network operator is a linear combination of tensors and/or tensor networks
        where each tensor (or tensor network) component attributes some of its open edges
        to the ket space and some to the dual bra space, thus establishing a map between
        the two. Therefore, a tensor network operator has a ket shape and a bra shape.
    (b) A tensor network operator may act on a ket tensor network expansion if its ket
        shape matches the shape of this ket tensor network expansion.
        A tensor network operator may act on a bra tensor network expansion if its bra
        shape matches the shape of this bra tensor network expansion.
    (c) A full contraction may be formed between a bra tensor network expansion,
        a tensor network operator, and a ket tensor network expansion if the bra shape
        of the tensor network operator matches the shape of the bra tensor network
        expansion and the ket shape of the tensor network operator matches the shape
        if the ket tensor network expansion.
    (d) Any contraction of a tensor network operator with a ket/bra tensor network
        expansion (or both) forms a tensor network expansion.
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


/** Scales a tensor by a scalar value. **/
template<typename NumericType>
inline bool scaleTensor(const std::string & name, //in: tensor name
                        NumericType value)        //in: scalar value
 {return numericalServer->scaleTensor(name,value);}

template<typename NumericType>
inline bool scaleTensorSync(const std::string & name, //in: tensor name
                            NumericType value)        //in: scalar value
 {return numericalServer->scaleTensorSync(name,value);}


/** Broadcast a tensor among all MPI processes within an intra-communicator.
    This function is needed when a tensor is updated in an operation
    submitted to a subset of MPI processes such that the excluded
    MPI processes can receive an updated version of the tensor. **/
inline bool broadcastTensor(const std::string & name, //in: tensor name
                            int root_process_rank,    //in: rank of the root process within the MPI intra-communicator
                            MPICommProxy intra_comm = MPICommProxy()) //in: MPI intra-communicator, defaults to all processes
 {return numericalServer->broadcastTensor(name,root_process_rank,intra_comm);}

inline bool broadcastTensorSync(const std::string & name, //in: tensor name
                                int root_process_rank,    //in: rank of the root process within the MPI intra-communicator
                                MPICommProxy intra_comm = MPICommProxy()) //in: MPI intra-communicator, defaults to all processes
 {return numericalServer->broadcastTensorSync(name,root_process_rank,intra_comm);}


/** Performs a global sum reduction on a tensor among all MPI processes within
    an intra-communicator. This function is needed when multiple MPI processes
    compute their local updates to the tensor, thus requiring a global sum
    reduction such that each MPI process will get the final (same) tensor value. **/
inline bool allreduceTensor(const std::string & name, //in: tensor name
                            MPICommProxy intra_comm = MPICommProxy()) //in: MPI intra-communicator, defaults to all processes
 {return numericalServer->allreduceTensor(name,intra_comm);}

inline bool allreduceTensorSync(const std::string & name, //in: tensor name
                                MPICommProxy intra_comm = MPICommProxy()) //in: MPI intra-communicator, defaults to all processes
 {return numericalServer->allreduceTensorSync(name,intra_comm);}


/** Transforms (updates) a tensor according to a user-defined tensor functor. **/
inline bool transformTensor(const std::string & name,              //in: tensor name
                            std::shared_ptr<TensorMethod> functor) //in: functor defining tensor transformation
 {return numericalServer->transformTensor(name,functor);}

inline bool transformTensorSync(const std::string & name,              //in: tensor name
                                std::shared_ptr<TensorMethod> functor) //in: functor defining tensor transformation
 {return numericalServer->transformTensorSync(name,functor);}


/** Extracts a slice from a tensor and stores it in another tensor
    the signature and shape of which determines which slice to extract. **/
inline bool extractTensorSlice(const std::string & tensor_name, //in: tensor name
                               const std::string & slice_name)  //in: slice name
 {return numericalServer->extractTensorSlice(tensor_name,slice_name);}

inline bool extractTensorSliceSync(const std::string & tensor_name, //in: tensor name
                                   const std::string & slice_name)  //in: slice name
 {return numericalServer->extractTensorSliceSync(tensor_name,slice_name);}


/** Inserts a slice into a tensor. The signature and shape of the slice
    determines the position in the tensor where the slice will be inserted. **/
inline bool insertTensorSlice(const std::string & tensor_name, //in: tensor name
                              const std::string & slice_name)  //in: slice name
 {return numericalServer->insertTensorSlice(tensor_name,slice_name);}

inline bool insertTensorSliceSync(const std::string & tensor_name, //in: tensor name
                                  const std::string & slice_name)  //in: slice name
 {return numericalServer->insertTensorSliceSync(tensor_name,slice_name);}


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


/** Decomposes a tensor into three tensor factors via SVD. The symbolic
    tensor contraction specification specifies the decomposition,
    for example:
     D(a,b,c,d,e) = L(c,i,e,j) * S(i,j) * R(b,j,a,i,d)
    where
     L(c,i,e,j) is the left SVD factor,
     R(b,j,a,i,d) is the right SVD factor,
     S(i,j) is the middle SVD factor (the diagonal with singular values).
    Note that the ordering of the contracted indices is not guaranteed! **/
inline bool decomposeTensorSVD(const std::string & contraction) //in: three-factor symbolic tensor contraction specification
 {return numericalServer->decomposeTensorSVD(contraction);}

inline bool decomposeTensorSVDSync(const std::string & contraction) //in: three-factor symbolic tensor contraction specification
 {return numericalServer->decomposeTensorSVDSync(contraction);}


/** Decomposes a tensor into two tensor factors via SVD. The symbolic
    tensor contraction specification specifies the decomposition,
    for example:
     D(a,b,c,d,e) = L(c,i,e,j) * R(b,j,a,i,d)
    where
     L(c,i,e,j) is the left SVD factor with absorbed singular values,
     R(b,j,a,i,d) is the right SVD factor. **/
inline bool decomposeTensorSVDL(const std::string & contraction) //in: two-factor symbolic tensor contraction specification
 {return numericalServer->decomposeTensorSVDL(contraction);}

inline bool decomposeTensorSVDLSync(const std::string & contraction) //in: two-factor symbolic tensor contraction specification
 {return numericalServer->decomposeTensorSVDLSync(contraction);}


/** Decomposes a tensor into two tensor factors via SVD. The symbolic
    tensor contraction specification specifies the decomposition,
    for example:
     D(a,b,c,d,e) = L(c,i,e,j) * R(b,j,a,i,d)
    where
     L(c,i,e,j) is the left SVD factor,
     R(b,j,a,i,d) is the right SVD factor with absorbed singular values. **/
inline bool decomposeTensorSVDR(const std::string & contraction) //in: two-factor symbolic tensor contraction specification
 {return numericalServer->decomposeTensorSVDR(contraction);}

inline bool decomposeTensorSVDRSync(const std::string & contraction) //in: two-factor symbolic tensor contraction specification
 {return numericalServer->decomposeTensorSVDRSync(contraction);}


/** Decomposes a tensor into two tensor factors via SVD. The symbolic
    tensor contraction specification specifies the decomposition,
    for example:
     D(a,b,c,d,e) = L(c,i,e,j) * R(b,j,a,i,d)
    where
     L(c,i,e,j) is the left SVD factor with absorbed square root of singular values,
     R(b,j,a,i,d) is the right SVD factor with absorbed square root of singular values. **/
inline bool decomposeTensorSVDLR(const std::string & contraction) //in: two-factor symbolic tensor contraction specification
 {return numericalServer->decomposeTensorSVDLR(contraction);}

inline bool decomposeTensorSVDLRSync(const std::string & contraction) //in: two-factor symbolic tensor contraction specification
 {return numericalServer->decomposeTensorSVDLRSync(contraction);}


/** Orthogonalizes a tensor by decomposing it via SVD while discarding
    the middle tensor factor with singular values. The symbolic tensor contraction
    specification specifies the decomposition. It must contain strictly one contracted index. **/
inline bool orthogonalizeTensorSVD(const std::string & contraction) //in: two-factor symbolic tensor contraction specification
 {return numericalServer->orthogonalizeTensorSVD(contraction);}

inline bool orthogonalizeTensorSVDSync(const std::string & contraction) //in: two-factor symbolic tensor contraction specification
 {return numericalServer->orthogonalizeTensorSVDSync(contraction);}


/** Orthogonalizes an isometric tensor over its isometric dimensions via the modified Gram-Schmidt procedure.  **/
inline bool orthogonalizeTensorMGS(const std::string & name) //in: tensor name
 {return numericalServer->orthogonalizeTensorMGS(name);}

inline bool orthogonalizeTensorMGSSync(const std::string & name) //in: tensor name
 {return numericalServer->orthogonalizeTensorMGSSync(name);}


/** Performs a full evaluation of a tensor network specified symbolically, based on
    the symbolic names of previously created tensors (including the output tensor). **/
inline bool evaluateTensorNetwork(const std::string & name,    //in: tensor network name
                                  const std::string & network) //in: symbolic tensor network specification
 {return numericalServer->evaluateTensorNetwork(name,network);}

inline bool evaluateTensorNetworkSync(const std::string & name,    //in: tensor network name
                                      const std::string & network) //in: symbolic tensor network specification
 {return numericalServer->evaluateTensorNetworkSync(name,network);}

inline bool evaluateTensorNetwork(const std::string & name,    //in: tensor network name
                                  const std::string & network, //in: symbolic tensor network specification
                                  const std::vector<unsigned int> & process_set) //in: chosen set of parallel processes (by their ranks)
 {return numericalServer->evaluateTensorNetwork(name,network,process_set);}

inline bool evaluateTensorNetworkSync(const std::string & name,    //in: tensor network name
                                      const std::string & network, //in: symbolic tensor network specification
                                      const std::vector<unsigned int> & process_set) //in: chosen set of parallel processes (by their ranks)
 {return numericalServer->evaluateTensorNetworkSync(name,network,process_set);}


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

inline bool evaluate(TensorNetwork & network, //in: finalized tensor network
                     const std::vector<unsigned int> & process_set) //in: chosen set of parallel processes (by their ranks)
 {return numericalServer->submit(network,process_set);}

inline bool evaluateSync(TensorNetwork & network, //in: finalized tensor network
                         const std::vector<unsigned int> & process_set) //in: chosen set of parallel processes (by their ranks)
 {bool success = numericalServer->submit(network,process_set);
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

inline bool evaluate(TensorExpansion & expansion,         //in: tensor network expansion
                     std::shared_ptr<Tensor> accumulator, //inout: tensor accumulator
                     const std::vector<unsigned int> & process_set) //in: chosen set of parallel processes (by their ranks)
 {return numericalServer->submit(expansion,accumulator,process_set);}

inline bool evaluateSync(TensorExpansion & expansion,         //in: tensor network expansion
                         std::shared_ptr<Tensor> accumulator, //inout: tensor accumulator
                         const std::vector<unsigned int> & process_set) //in: chosen set of parallel processes (by their ranks)
 {if(!accumulator) return false;
  bool success = numericalServer->submit(expansion,accumulator,process_set);
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


/** Resets the tensor contraction sequence optimizer that
    is invoked when evaluating tensor networks: {dummy,heuro}. **/
inline void resetContrSeqOptimizer(const std::string & optimizer_name)
 {return numericalServer->resetContrSeqOptimizer(optimizer_name);}


/** Resets tensor runtime logging level (0:none). **/
inline void resetRuntimeLoggingLevel(int level = 0)
 {return numericalServer->resetRuntimeLoggingLevel(level);}

} //namespace exatn

#endif //EXATN_NUMERICS_HPP_
