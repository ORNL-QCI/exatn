/** ExaTN::Numerics: General client header
REVISION: 2021/04/01

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

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
        an index label can only refer to a single registered (named) subspace it is associated with.
 3. Tensor:
    (a) A tensor is defined by its name, shape and signature.
    (b) Tensor shape is an ordered tuple of tensor dimension extents.
    (c) Tensor signature is an ordered tuple of {space_id,subspace_id} pairs
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
        Partial/full traces within a tensor are allowed in a tensor network although
        they must be supported by the processing backend in order to be computed.
    (b) The same tensor may be present in the given tensor network multiple times
        via different vertices, either normal or conjugated.
    (c) Each tensor network has an implicit output tensor collecting all open
        edges from all input tensors. Evaluating the tensor network means
        computing the value of this output tensor, given all input tensors.
    (d) The conjugation operation applied to a tensor network performs complex conjugation
        of all constituent input tensors, but does not apply to the output tensor per se
        because the output tensor is simply the result of the full contraction of the tensor
        network. The conjugation operation also reverses the direction of all edges.
    (e) An input tensor may be present in multiple tensor networks and its lifetime
        is not bound to the lifetime of any tensor network it belongs to.
 6. Tensor network expansion:
    (a) Tensor network expansion is a linear combination of tensor networks
        with some complex coefficients. The output tensors of all constituent
        tensor networks must be congruent. Evaluating the tensor network
        expansion means computing the sum of all these output tensors scaled
        by their respective (complex) coefficients.
    (b) A tensor network expansion may either belong to the primary ket or dual bra space.
        The conjugation operation transitions the tensor network expansion between
        the ket and bra spaces.
    (c) A single tensor network may enter multiple tensor network expansions and its
        lifetime is not bound by the lifetime of any tensor network expansion it belongs to.
 7. Tensor network operator:
    (a) Tensor network operator is a linear combination of tensors and/or tensor networks
        where each tensor (or tensor network) component associates some of its open edges
        to a primary ket space and some to a dual bra space, thus establishing a map
        between the two generally distinct spaces. Therefore, a tensor network operator
        has a ket shape and a bra shape. All components of a tensor network operator
        must adhere to the same ket/bra shapes.
    (b) A tensor network operator may act on a ket tensor network expansion if its ket
        shape matches the shape of that ket tensor network expansion.
        A tensor network operator may act on a bra tensor network expansion if its bra
        shape matches the shape of that bra tensor network expansion.
    (c) A full contraction may be formed between a bra tensor network expansion,
        a tensor network operator, and a ket tensor network expansion if the bra shape
        of the tensor network operator matches the shape of the bra tensor network
        expansion and the ket shape of the tensor network operator matches the shape
        of the ket tensor network expansion.
    (d) Any contraction of a tensor network operator with a ket/bra tensor network
        expansion (or both) forms another tensor network expansion.
**/

#ifndef EXATN_NUMERICS_HPP_
#define EXATN_NUMERICS_HPP_

#include "num_server.hpp"

#include <utility>
#include <memory>
#include <string>

namespace exatn{

////////////////////////
// TAProL SCOPING API //
////////////////////////

/** Opens a new (child) TAProL scope and returns its id. **/
inline ScopeId openScope(const std::string & scope_name) //new scope name
 {return numericalServer->openScope(scope_name);}


/** Closes the currently open TAProL scope and returns its parental scope id. **/
inline ScopeId closeScope()
 {return numericalServer->closeScope();}


/////////////////////////////////////
// SPACE/SUBSPACE REGISTRATION API //
/////////////////////////////////////

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


///////////////////////////////////////////
// EXTERNAL METHOD/DATA REGISTRATION API //
///////////////////////////////////////////

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


///////////////////////
// TENSOR HELPER API //
///////////////////////

/** Checks whether a given tensor has been allocated storage (created). **/
inline bool tensorAllocated(const std::string & name) //in: tensor name
 {return numericalServer->tensorAllocated(name);}


/** Returns a shared pointer to the actual tensor object. **/
inline std::shared_ptr<Tensor> getTensor(const std::string & name) //in: tensor name
 {return numericalServer->getTensor(name);}


/** Returns the reference to the actual tensor object. **/
inline Tensor & getTensorRef(const std::string & name) //in: tensor name
 {return numericalServer->getTensorRef(name);}


/** Returns the tensor element type (if allocated storage). **/
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


//////////////////////////
// TENSOR OPERATION API //
//////////////////////////

/** Declares, registers and actually creates a tensor via the processing backend.
    See numerics::Tensor constructors for different creation options. **/
template <typename... Args>
inline bool createTensor(const std::string & name,       //in: tensor name
                         TensorElementType element_type, //in: tensor element type
                         Args&&... args)                 //in: other arguments for Tensor ctor
 {return numericalServer->createTensor(name,element_type,std::forward<Args>(args)...);}

template <typename... Args>
inline bool createTensorSync(const std::string & name,       //in: tensor name
                             TensorElementType element_type, //in: tensor element type
                             Args&&... args)                 //in: other arguments for Tensor ctor
 {return numericalServer->createTensorSync(name,element_type,std::forward<Args>(args)...);}

inline bool createTensor(std::shared_ptr<Tensor> tensor, //in: existing declared tensor (can be composite)
                         TensorElementType element_type) //in: tensor element type
 {return numericalServer->createTensor(tensor,element_type);}

inline bool createTensorSync(std::shared_ptr<Tensor> tensor, //in: existing declared tensor (can be composite)
                             TensorElementType element_type) //in: tensor element type
 {return numericalServer->createTensorSync(tensor,element_type);}

inline bool createTensor(const std::string & name,          //in: tensor name
                         const TensorSignature & signature, //in: tensor signature with registered spaces/subspaces
                         TensorElementType element_type)    //in: tensor element type
 {return numericalServer->createTensor(name,signature,element_type);}

template <typename... Args>
inline bool createTensor(const ProcessGroup & process_group, //in: chosen group of MPI processes
                         const std::string & name,           //in: tensor name
                         TensorElementType element_type,     //in: tensor element type
                         Args&&... args)                     //in: other arguments for Tensor ctor
 {return numericalServer->createTensor(process_group,name,element_type,std::forward<Args>(args)...);}

template <typename... Args>
inline bool createTensorSync(const ProcessGroup & process_group, //in: chosen group of MPI processes
                             const std::string & name,           //in: tensor name
                             TensorElementType element_type,     //in: tensor element type
                             Args&&... args)                     //in: other arguments for Tensor ctor
 {return numericalServer->createTensorSync(process_group,name,element_type,std::forward<Args>(args)...);}

inline bool createTensor(const ProcessGroup & process_group, //in: chosen group of MPI processes
                         std::shared_ptr<Tensor> tensor,     //in: existing declared tensor (can be composite)
                         TensorElementType element_type)     //in: tensor element type
 {return numericalServer->createTensor(process_group,tensor,element_type);}

inline bool createTensorSync(const ProcessGroup & process_group, //in: chosen group of MPI processes
                             std::shared_ptr<Tensor> tensor,     //in: existing declared tensor (can be composite)
                             TensorElementType element_type)     //in: tensor element type
 {return numericalServer->createTensorSync(process_group,tensor,element_type);}


/** Creates and allocates storage for a composite tensor distributed over a given process group.
    The distribution of tensor blocks is dictated by its split dimensions. **/
template <typename... Args>
inline bool createTensor(const ProcessGroup & process_group,                      //in: chosen group of MPI processes
                         const std::string & name,                                //in: tensor name
                         const std::vector<std::pair<unsigned int,
                                                     unsigned int>> & split_dims, //in: split tensor dimensions: pair{Dimension,MaxDepth}
                         TensorElementType element_type,                          //in: tensor element type
                         Args&&... args)                                          //in: other arguments for Tensor ctor
 {return numericalServer->createTensor(process_group,name,split_dims,element_type,std::forward<Args>(args)...);}

template <typename... Args>
inline bool createTensorSync(const ProcessGroup & process_group,                      //in: chosen group of MPI processes
                             const std::string & name,                                //in: tensor name
                             const std::vector<std::pair<unsigned int,
                                                         unsigned int>> & split_dims, //in: split tensor dimensions: pair{Dimension,MaxDepth}
                             TensorElementType element_type,                          //in: tensor element type
                             Args&&... args)                                          //in: other arguments for Tensor ctor
 {return numericalServer->createTensorSync(process_group,name,split_dims,element_type,std::forward<Args>(args)...);}


/** Creates all tensors in a given tensor network that are still unallocated storage. **/
inline bool createTensors(TensorNetwork & tensor_network,         //inout: tensor network
                          TensorElementType element_type)         //in: tensor element type
 {return numericalServer->createTensors(tensor_network,element_type);}

inline bool createTensorsSync(TensorNetwork & tensor_network,     //inout: tensor network
                              TensorElementType element_type)     //in: tensor element type
 {return numericalServer->createTensorsSync(tensor_network,element_type);}

inline bool createTensors(const ProcessGroup & process_group,     //in: chosen group of MPI processes
                          TensorNetwork & tensor_network,         //inout: tensor network
                          TensorElementType element_type)         //in: tensor element type
 {return numericalServer->createTensors(process_group,tensor_network,element_type);}

inline bool createTensorsSync(const ProcessGroup & process_group, //in: chosen group of MPI processes
                              TensorNetwork & tensor_network,     //inout: tensor network
                              TensorElementType element_type)     //in: tensor element type
 {return numericalServer->createTensorsSync(process_group,tensor_network,element_type);}


/** Destroys a tensor, including its backend representation. **/
inline bool destroyTensor(const std::string & name) //in: tensor name
 {return numericalServer->destroyTensor(name);}

inline bool destroyTensorSync(const std::string & name) //in: tensor name
 {return numericalServer->destroyTensorSync(name);}


/** Destroys all currently allocated tensors in a given tensor network.
    Note that the destroyed tensors could also be present in other tensor networks
    in which case they will no longer be processible. **/
inline bool destroyTensors(TensorNetwork & tensor_network)     //inout: tensor network
 {return numericalServer->destroyTensors(tensor_network);}

inline bool destroyTensorsSync(TensorNetwork & tensor_network) //inout: tensor network
 {return numericalServer->destroyTensorsSync(tensor_network);}


/** Initializes a tensor to some scalar value. **/
template<typename NumericType>
inline bool initTensor(const std::string & name, //in: tensor name
                       NumericType value)        //in: scalar value
 {return numericalServer->initTensor(name,value);}

template<typename NumericType>
inline bool initTensorSync(const std::string & name, //in: tensor name
                           NumericType value)        //in: scalar value
 {return numericalServer->initTensorSync(name,value);}


/** Initializes a tensor with externally provided data stored
    in a flattened vector in the generalized column-wise order. **/
template<typename NumericType>
inline bool initTensorData(const std::string & name,                  //in: tensor name
                           const std::vector<NumericType> & ext_data) //in: externally provided data
 {return numericalServer->initTensorData(name,ext_data);}

template<typename NumericType>
inline bool initTensorDataSync(const std::string & name,                  //in: tensor name
                               const std::vector<NumericType> & ext_data) //in: externally provided data
 {return numericalServer->initTensorDataSync(name,ext_data);}


/** Initializes a tensor with externally provided data read from a file with format:
     Storage format (string: {dense|list})
     Tensor name
     Tensor shape (space-separated dimension extents)
     Tensor signature (space-separated dimension base offsets)
     Tensor elements:
      Dense format: Numeric values (column-wise order), any number of values per line;
      List format: Numeric value and Multi-index in each line. **/
inline bool initTensorFile(const std::string & name,     //in: tensor name
                           const std::string & filename) //in: file name with tensor data
 {return numericalServer->initTensorFile(name,filename);}

inline bool initTensorFileSync(const std::string & name,     //in: tensor name
                               const std::string & filename) //in: file name with tensor data
 {return numericalServer->initTensorFileSync(name,filename);}


/** Initializes the tensor body with random values. **/
inline bool initTensorRnd(const std::string & name) //in: tensor name
 {return numericalServer->initTensorRnd(name);}

inline bool initTensorRndSync(const std::string & name) //in: tensor name
 {return numericalServer->initTensorRndSync(name);}


/** Initializes all input tensors of a given tensor network to a random value. **/
inline bool initTensorsRnd(TensorNetwork & tensor_network)     //inout: tensor network
 {return numericalServer->initTensorsRnd(tensor_network);}

inline bool initTensorsRndSync(TensorNetwork & tensor_network) //inout: tensor network
 {return numericalServer->initTensorsRndSync(tensor_network);}


/** Computes max-abs norm of a tensor. **/
inline bool computeMaxAbsSync(const std::string & name, //in: tensor name
                              double & norm)            //out: tensor norm
 {return numericalServer->computeMaxAbsSync(name,norm);}


/** Computes 1-norm of a tensor. **/
inline bool computeNorm1Sync(const std::string & name, //in: tensor name
                             double & norm)            //out: tensor norm
 {return numericalServer->computeNorm1Sync(name,norm);}


/** Computes 2-norm of a tensor. **/
inline bool computeNorm2Sync(const std::string & name, //in: tensor name
                             double & norm)            //out: tensor norm
 {return numericalServer->computeNorm2Sync(name,norm);}


/** Computes partial 2-norms over a chosen tensor dimension. **/
inline bool computePartialNormsSync(const std::string & name,            //in: tensor name
                                    unsigned int tensor_dimension,       //in: chosen tensor dimension
                                    std::vector<double> & partial_norms) //out: partial 2-norms over the chosen tensor dimension
 {return numericalServer->computePartialNormsSync(name,tensor_dimension,partial_norms);}


/** Replicates a tensor within the given process group, which defaults to all MPI processes.
    Only the root_process_rank within the given process group is required to have the tensor,
    that is, the tensor will automatically be created in those MPI processes which do not have it.  **/
inline bool replicateTensor(const std::string & name,           //in: tensor name
                            int root_process_rank)              //in: local rank of the root process within the given process group
 {return numericalServer->replicateTensor(name,root_process_rank);}

inline bool replicateTensorSync(const std::string & name,       //in: tensor name
                                int root_process_rank)          //in: local rank of the root process within the given process group
 {return numericalServer->replicateTensorSync(name,root_process_rank);}

inline bool replicateTensor(const ProcessGroup & process_group, //in: chosen group of MPI processes
                            const std::string & name,           //in: tensor name
                            int root_process_rank)              //in: local rank of the root process within the given process group
 {return numericalServer->replicateTensor(process_group,name,root_process_rank);}

inline bool replicateTensorSync(const ProcessGroup & process_group, //in: chosen group of MPI processes
                                const std::string & name,           //in: tensor name
                                int root_process_rank)              //in: local rank of the root process within the given process group
 {return numericalServer->replicateTensorSync(process_group,name,root_process_rank);}


/** Broadcast a tensor among all MPI processes within a given process group,
    which defaults to all MPI processes. This function is needed when
    a tensor is updated in an operation submitted to a subset of MPI processes
    such that the excluded MPI processes can receive an updated version of the tensor.
    Note that the tensor must exist in all participating MPI processes. **/
inline  bool broadcastTensor(const std::string & name,          //in: tensor name
                             int root_process_rank)             //in: local rank of the root process within the given process group
 {return numericalServer->broadcastTensor(name,root_process_rank);}

inline bool broadcastTensorSync(const std::string & name,       //in: tensor name
                                int root_process_rank)          //in: local rank of the root process within the given process group
 {return numericalServer->broadcastTensorSync(name,root_process_rank);}

inline bool broadcastTensor(const ProcessGroup & process_group, //in: chosen group of MPI processes
                            const std::string & name,           //in: tensor name
                            int root_process_rank)              //in: local rank of the root process within the given process group
 {return numericalServer->broadcastTensor(process_group,name,root_process_rank);}

inline bool broadcastTensorSync(const ProcessGroup & process_group, //in: chosen group of MPI processes
                                const std::string & name,           //in: tensor name
                                int root_process_rank)              //in: local rank of the root process within the given process group
 {return numericalServer->broadcastTensorSync(process_group,name,root_process_rank);}


/** Performs a global sum reduction on a tensor among all MPI processes within a given
    process group, which defaults to all MPI processes. This function is needed when
    multiple MPI processes compute their local updates to the tensor, thus requiring
    a global sum reduction such that each MPI process will get the final (same) tensor
    value. Note that the tensor must exist in all participating MPI processes. **/
inline bool allreduceTensor(const std::string & name)           //in: tensor name
 {return numericalServer->allreduceTensor(name);}

inline bool allreduceTensorSync(const std::string & name)       //in: tensor name
 {return numericalServer->allreduceTensorSync(name);}

inline bool allreduceTensor(const ProcessGroup & process_group, //in: chosen group of MPI processes
                            const std::string & name)           //in: tensor name
 {return numericalServer->allreduceTensor(process_group,name);}

inline bool allreduceTensorSync(const ProcessGroup & process_group, //in: chosen group of MPI processes
                                const std::string & name)           //in: tensor name
 {return numericalServer->allreduceTensorSync(process_group,name);}


/** Scales a tensor by a scalar value. **/
template<typename NumericType>
inline bool scaleTensor(const std::string & name,       //in: tensor name
                        NumericType value)              //in: scalar value
 {return numericalServer->scaleTensor(name,value);}

template<typename NumericType>
inline bool scaleTensorSync(const std::string & name,   //in: tensor name
                            NumericType value)          //in: scalar value
 {return numericalServer->scaleTensorSync(name,value);}


/** Transforms (updates) a tensor according to a user-defined tensor functor. **/
inline bool transformTensor(const std::string & name,                  //in: tensor name
                            std::shared_ptr<TensorMethod> functor)     //in: functor defining the tensor transformation
 {return numericalServer->transformTensor(name,functor);}

inline bool transformTensorSync(const std::string & name,              //in: tensor name
                                std::shared_ptr<TensorMethod> functor) //in: functor defining the tensor transformation
 {return numericalServer->transformTensorSync(name,functor);}

inline bool transformTensor(const std::string & name,                  //in: tensor name
                            const std::string & functor_name)          //in: name of the functor defining the tensor transformation
 {return numericalServer->transformTensor(name,functor_name);}

inline bool transformTensorSync(const std::string & name,              //in: tensor name
                                const std::string & functor_name)      //in: name of the functor defining the tensor transformation
 {return numericalServer->transformTensorSync(name,functor_name);}


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


/** Prints a tensor to the standard output. **/
inline bool printTensor(const std::string & name) //in: tensor name
 {return numericalServer->printTensor(name);}

inline bool printTensorSync(const std::string & name) //in: tensor name
 {return numericalServer->printTensorSync(name);}


/** Prints a tensor to a file. **/
inline bool printTensorFile(const std::string & name,     //in: tensor name
                            const std::string & filename) //in: file name
 {return numericalServer->printTensorFile(name,filename);}

inline bool printTensorFileSync(const std::string & name,     //in: tensor name
                                const std::string & filename) //in: file name
 {return numericalServer->printTensorFileSync(name,filename);}


/** Performs a full evaluation of a tensor network specified symbolically, based on
    the symbolic names of previously created tensors (including the output tensor). **/
inline bool evaluateTensorNetwork(const std::string & name,    //in: tensor network name
                                  const std::string & network) //in: symbolic tensor network specification
 {return numericalServer->evaluateTensorNetwork(name,network);}

inline bool evaluateTensorNetworkSync(const std::string & name,    //in: tensor network name
                                      const std::string & network) //in: symbolic tensor network specification
 {return numericalServer->evaluateTensorNetworkSync(name,network);}

inline bool evaluateTensorNetwork(const ProcessGroup & process_group, //in: chosen group of MPI processes
                                  const std::string & name,           //in: tensor network name
                                  const std::string & network)        //in: symbolic tensor network specification
 {return numericalServer->evaluateTensorNetwork(process_group,name,network);}

inline bool evaluateTensorNetworkSync(const ProcessGroup & process_group, //in: chosen group of MPI processes
                                      const std::string & name,           //in: tensor network name
                                      const std::string & network)        //in: symbolic tensor network specification
 {return numericalServer->evaluateTensorNetworkSync(process_group,name,network);}


/** Evaluates a tensor network object (computes the output tensor). **/
inline bool evaluate(TensorNetwork & network) //in: finalized tensor network
 {return numericalServer->submit(network);}

inline bool evaluateSync(TensorNetwork & network) //in: finalized tensor network
 {bool success = numericalServer->submit(network);
  if(success) success = numericalServer->sync(network);
  return success;}

inline bool evaluate(const ProcessGroup & process_group, //in: chosen group of MPI processes
                     TensorNetwork & network)            //in: finalized tensor network
 {return numericalServer->submit(process_group,network);}

inline bool evaluateSync(const ProcessGroup & process_group, //in: chosen group of MPI processes
                         TensorNetwork & network)            //in: finalized tensor network
 {bool success = numericalServer->submit(process_group,network);
  if(success) success = numericalServer->sync(process_group,network);
  return success;}


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

inline bool evaluate(const ProcessGroup & process_group,  //in: chosen group of MPI processes
                     TensorExpansion & expansion,         //in: tensor network expansion
                     std::shared_ptr<Tensor> accumulator) //inout: tensor accumulator
 {return numericalServer->submit(process_group,expansion,accumulator);}

inline bool evaluateSync(const ProcessGroup & process_group,  //in: chosen group of MPI processes
                         TensorExpansion & expansion,         //in: tensor network expansion
                         std::shared_ptr<Tensor> accumulator) //inout: tensor accumulator
 {if(!accumulator) return false;
  bool success = numericalServer->submit(process_group,expansion,accumulator);
  if(success) success = numericalServer->sync(process_group,*accumulator);
  return success;}


/** Synchronizes all outstanding update operations on a given tensor specified by
    its symbolic name. If ProcessGroup is not provided, defaults to the local process.**/
inline bool sync(const std::string & name, //in: tensor name
                 bool wait = true)         //in: wait versus test for completion
 {return numericalServer->sync(name,wait);}

inline bool sync(const ProcessGroup & process_group, //in: chosen group of MPI processes
                 const std::string & name,           //in: tensor name
                 bool wait = true)                   //in: wait versus test for completion
 {return numericalServer->sync(process_group,name,wait);}


/** Synchronizes all outstanding operations on a given tensor.
    If ProcessGroup is not provided, defaults to the local process. **/
inline bool sync(const Tensor & tensor, //in: tensor
                 bool wait = true)      //in: wait versus test for completion
 {return numericalServer->sync(tensor,wait);}

inline bool sync(const ProcessGroup & process_group,  //in: chosen group of MPI processes
                 const Tensor & tensor,               //in: tensor
                 bool wait = true)                    //in: wait versus test for completion
 {return numericalServer->sync(process_group,tensor,wait);}


/** Synchronizes all outstanding operations on a given tensor network object.
    If ProcessGroup is not provided, defaults to the local process. **/
inline bool sync(TensorNetwork & network, //in: finalized tensor network
                 bool wait = true)        //in: wait versus test for completion
 {return numericalServer->sync(network,wait);}

inline bool sync(const ProcessGroup & process_group, //in: chosen group of MPI processes
                 TensorNetwork & network,            //in: finalized tensor network
                 bool wait = true)                   //in: wait versus test for completion
 {return numericalServer->sync(process_group,network,wait);}


/** Synchronizes all outstanding tensor operations in the current scope (barrier).
    If ProcessGroup is not provided, defaults to the local process. **/
inline bool sync(bool wait = true)                    //in: wait versus test for completion
 {return numericalServer->sync(wait);}

inline bool sync(const ProcessGroup & process_group,  //in: chosen group of MPI processes
                 bool wait = true)                    //in: wait versus test for completion
 {return numericalServer->sync(process_group,wait);}


/** Normalizes a tensor to a given 2-norm. **/
inline bool normalizeNorm2Sync(const std::string & name, //in: tensor name
                               double norm = 1.0)        //in: desired 2-norm
 {return numericalServer->normalizeNorm2Sync(name,norm);}


/** Normalizes a tensor network expansion to a given 2-norm by rescaling
    all tensor network components by the same factor: Only the tensor
    network expansion coeffcients are affected, not tensors. **/
inline bool normalizeNorm2Sync(TensorExpansion & expansion, //inout: tensor network expansion
                               double norm = 1.0)           //in: desired 2-norm
 {return numericalServer->normalizeNorm2Sync(expansion,norm);}

inline bool normalizeNorm2Sync(const ProcessGroup & process_group, //in: chosen group of MPI processes
                               TensorExpansion & expansion,        //inout: tensor network expansion
                               double norm = 1.0)                  //in: desired 2-norm
 {return numericalServer->normalizeNorm2Sync(process_group,expansion,norm);}


/** Normalizes all input tensors in a tensor network to a given 2-norm.
    If only_optimizable is TRUE, only optimizable tensors will be normalized. **/
inline bool balanceNorm2Sync(TensorNetwork & network,       //inout: tensor network
                             double norm = 1.0,             //in: desired 2-norm
                             bool only_optimizable = false) //in: whether to normalize only optimizable tensors
 {return numericalServer->balanceNorm2Sync(network,norm,only_optimizable);}

inline bool balanceNorm2Sync(const ProcessGroup & process_group, //in: chosen group of MPI processes
                             TensorNetwork & network,            //inout: tensor network
                             double norm = 1.0,                  //in: desired 2-norm
                             bool only_optimizable = false)      //in: whether to normalize only optimizable tensors
 {return numericalServer->balanceNorm2Sync(process_group,network,norm,only_optimizable);}


/** Normalizes all input tensors in a tensor network expansion to a given 2-norm.
    If only_optimizable is TRUE, only optimizable tensors will be normalized. **/
inline bool balanceNorm2Sync(TensorExpansion & expansion,   //inout: tensor network expansion
                             double norm = 1.0,             //in: desired 2-norm
                             bool only_optimizable = false) //in: whether to normalize only optimizable tensors
 {return numericalServer->balanceNorm2Sync(expansion,norm,only_optimizable);}

inline bool balanceNorm2Sync(const ProcessGroup & process_group, //in: chosen group of MPI processes
                             TensorExpansion & expansion,        //inout: tensor network expansion
                             double norm = 1.0,                  //in: desired 2-norm
                             bool only_optimizable = false)      //in: whether to normalize only optimizable tensors
 {return numericalServer->balanceNorm2Sync(process_group,expansion,norm,only_optimizable);}


/** Normalizes all input tensors in a tensor network expansion to a given 2-norm
    and rescales tensor network expansion coefficients to normalize the entire
    tensor network expansion to another given 2-norm. If only_optimizable is TRUE,
    only optimizable tensors will be normalized. **/
inline bool balanceNormalizeNorm2Sync(TensorExpansion & expansion,   //inout: tensor network expansion
                                      double tensor_norm = 1.0,      //in: desired 2-norm of each input tensor
                                      double expansion_norm = 1.0,   //in: desired 2-norm of the tensor network expansion
                                      bool only_optimizable = false) //in: whether to normalize only optimizable tensors
 {return numericalServer->balanceNormalizeNorm2Sync(expansion,tensor_norm,expansion_norm,only_optimizable);}

inline bool balanceNormalizeNorm2Sync(const ProcessGroup & process_group, //in: chosen group of MPI processes
                                      TensorExpansion & expansion,        //inout: tensor network expansion
                                      double tensor_norm = 1.0,           //in: desired 2-norm of each input tensor
                                      double expansion_norm = 1.0,        //in: desired 2-norm of the tensor network expansion
                                      bool only_optimizable = false)      //in: whether to normalize only optimizable tensors
 {return numericalServer->balanceNormalizeNorm2Sync(process_group,expansion,tensor_norm,expansion_norm,only_optimizable);}


///////////////////////
// TENSOR ACCESS API //
///////////////////////

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


//////////////////////////////
// MISCELLENEOUS HELPER API //
//////////////////////////////

/** Prints a tensor contraction sequence. **/
inline void printContractionSequence(const std::list<numerics::ContrTriple> & contr_seq) //in: tensor contraction sequence
 {return numerics::printContractionSequence(contr_seq);}

inline void printContractionSequence(std::ofstream & output_file,                        //in: output file stream
                                     const std::list<numerics::ContrTriple> & contr_seq) //in: tensor contraction sequence
 {return numerics::printContractionSequence(output_file,contr_seq);}


/** Creates and returns a tensor network builder. **/
inline std::unique_ptr<exatn::NetworkBuilder> getTensorNetworkBuilder(const std::string & builder_name) //in: tensor network builder name
 {return std::move(exatn::NetworkBuildFactory::get()->createNetworkBuilder(builder_name));}


/** Constructs a tensor network from a symbolic specification.
    All participating tensors must have been created before. **/
inline std::shared_ptr<exatn::TensorNetwork> makeTensorNetwork(const std::string & name,     //in: tensor network name
                                                               const std::string & symbolic) //in: tensor network specification
 {std::vector<std::string> symb_tensors;
  auto success = parse_tensor_network(symbolic,symb_tensors); assert(success);
  std::map<std::string,std::shared_ptr<Tensor>> tensors;
  for(const auto & symb_tensor: symb_tensors){
   bool conj;
   std::string tens_name;
   std::vector<exatn::IndexLabel> indices;
   success = parse_tensor(symb_tensor,tens_name,indices,conj); assert(success);
   auto tensor = exatn::getTensor(tens_name);
   if(tensor){
    tensors.emplace(tens_name,tensor);
   }else{
    std::cout << "#ERROR(exatn::makeTensorNetwork): Tensor " << tens_name << " does not exist!" << std::endl;
    assert(false);
   }
  }
  return std::make_shared<TensorNetwork>(name,symbolic,tensors);
 }


//////////////////////////
// INTERNAL CONTROL API //
//////////////////////////

/** Resets the tensor contraction sequence optimizer that is invoked
    when evaluating tensor networks: {dummy,heuro,greed,metis}. **/
inline void resetContrSeqOptimizer(const std::string & optimizer_name)
 {return numericalServer->resetContrSeqOptimizer(optimizer_name);}


/** Activates optimized tensor contraction sequence caching for later reuse. **/
inline void activateContrSeqCaching(bool persist = false)
 {return numericalServer->activateContrSeqCaching(persist);}


/** Deactivates optimized tensor contraction sequence caching. **/
inline void deactivateContrSeqCaching()
 {return numericalServer->deactivateContrSeqCaching();}


/** Queries the status of optimized tensor contraction sequence caching. **/
inline bool queryContrSeqCaching()
 {return numericalServer->queryContrSeqCaching();}


/** Resets client logging level (0:none). **/
inline void resetClientLoggingLevel(int level = 0)
 {return numericalServer->resetClientLoggingLevel(level);}


/** Resets tensor runtime logging level (0:none). **/
inline void resetRuntimeLoggingLevel(int level = 0)
 {return numericalServer->resetRuntimeLoggingLevel(level);}


/** Resets both client and runtime logging level (0:none). **/
inline void resetLoggingLevel(int client_level = 0,
                              int runtime_level = 0)
 {resetClientLoggingLevel(client_level);
  resetRuntimeLoggingLevel(runtime_level);
  return;
 }


/** Resets tensor operation execution serialization. **/
inline void resetExecutionSerialization(bool serialize,
                                        bool validation_trace = false)
 {return numericalServer->resetExecutionSerialization(serialize,validation_trace);}


/** Activates/deactivates dry run (no actual computations). **/
inline void activateDryRun(bool dry_run)
 {return numericalServer->activateDryRun(dry_run);}


/** Activates mixed-precision fast math operations on all devices (if available). **/
inline void activateFastMath()
 {return numericalServer->activateFastMath();}


/** Returns the Host memory buffer size in bytes provided by the runtime. **/
inline std::size_t getMemoryBufferSize()
 {return numericalServer->getMemoryBufferSize();}


/** Returns the current value of the Flop counter. **/
inline double getTotalFlopCount()
 {return numericalServer->getTotalFlopCount();}


/** Returns the default process group comprising all MPI processes and their communicator. **/
inline const ProcessGroup & getDefaultProcessGroup()
 {return numericalServer->getDefaultProcessGroup();}


/** Returns the current process group comprising solely the current MPI process and its own communicator. **/
inline const ProcessGroup & getCurrentProcessGroup()
 {return numericalServer->getCurrentProcessGroup();}


/** Returns the global rank of the current MPI process in the default process group. **/
inline int getProcessRank()
 {return numericalServer->getProcessRank();}


/** Returns the total number of MPI processes in the default process group. **/
inline int getNumProcesses()
 {return numericalServer->getNumProcesses();}

} //namespace exatn

#endif //EXATN_NUMERICS_HPP_
