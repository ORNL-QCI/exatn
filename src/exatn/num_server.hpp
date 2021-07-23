/** ExaTN::Numerics: Numerical server
REVISION: 2021/07/23

Copyright (C) 2018-2021 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2021 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Numerical server provides basic tensor network processing functionality:
     + Opening/closing TAProL scopes (top scope 0 "GLOBAL" is open automatically);
     + Creation/destruction of named vector spaces and their named subspaces;
     + Registration/retrieval of external data (class BytePacket);
     + Registration/retrieval of external tensor methods (class TensorFunctor);
     + Submission for processing of individual tensor operations, tensor networks
       and tensor network expansions;
     + Higher-level methods for tensor creation, destruction, and operations on them,
       which use symbolic tensor names for tensor identification and processing.
 (b) Processing of individual tensor operations, tensor networks and tensor network expansions
     has asynchronous semantics: Submit TensorOperation/TensorNetwork/TensorExpansion for
     processing, then synchronize on the tensor-result or the accumulator tensor.
     Processing of a tensor operation means evaluating all its output tensor operands.
     Processing of a tensor network means evaluating its output tensor (#0),
     which is automatically initialized to zero before the evaluation.
     Processing of a tensor network expansion means evaluating all constituent
     tensor network components and accumulating them into the accumulator tensor
     with their respective prefactors. Synchronization of processing of a tensor operation,
     tensor network or tensor network expansion means ensuring that the tensor-result,
     either the output tensor or the accumulator tensor, has been fully computed.
 (c) Namespace exatn introduces a number of aliases for types imported from exatn::numerics.
     Importantly exatn::TensorMethod, which is talsh::TensorFunctor<Indentifiable>,
     defines the interface which needs to be implemented by the application in order
     to perform a custom unary transformation/initialization operation on exatn::Tensor.
     This is the only portable way to modify the tensor content in a desired way.
**/

#ifndef EXATN_NUM_SERVER_HPP_
#define EXATN_NUM_SERVER_HPP_

#include "tensor_basic.hpp"
#include "space_register.hpp"
#include "tensor_range.hpp"
#include "tensor.hpp"
#include "tensor_composite.hpp"
#include "tensor_operation.hpp"
#include "tensor_op_factory.hpp"
#include "tensor_symbol.hpp"
#include "tensor_network.hpp"
#include "tensor_operator.hpp"
#include "tensor_expansion.hpp"
#include "network_build_factory.hpp"
#include "contraction_seq_optimizer_factory.hpp"

#include "tensor_runtime.hpp"

#include "Identifiable.hpp"
#include "tensor_method.hpp"
#include "functor_init_val.hpp"
#include "functor_init_rnd.hpp"
#include "functor_init_dat.hpp"
#include "functor_init_delta.hpp"
#include "functor_init_proj.hpp"
#include "functor_init_file.hpp"
#include "functor_scale.hpp"
#include "functor_maxabs.hpp"
#include "functor_norm1.hpp"
#include "functor_norm2.hpp"
#include "functor_diag_rank.hpp"
#include "functor_print.hpp"

#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <stack>
#include <list>
#include <map>

#include "errors.hpp"

using exatn::Identifiable;

namespace exatn{

//Primary numerics:: types exposed to the user:
using numerics::VectorSpace;
using numerics::Subspace;
using numerics::TensorRange;
using numerics::TensorShape;
using numerics::TensorSignature;
using numerics::TensorLeg;
using numerics::Tensor;
using numerics::TensorComposite;
using numerics::TensorOperation;
using numerics::TensorOpFactory;
using numerics::TensorNetwork;
using numerics::TensorOperator;
using numerics::TensorExpansion;

using numerics::NetworkBuilder;
using numerics::NetworkBuildFactory;

using numerics::ContractionSeqOptimizer;
using numerics::ContractionSeqOptimizerFactory;

using numerics::FunctorInitVal;
using numerics::FunctorInitRnd;
using numerics::FunctorInitDat;
using numerics::FunctorInitDelta;
using numerics::FunctorInitProj;
using numerics::FunctorInitFile;
using numerics::FunctorScale;
using numerics::FunctorMaxAbs;
using numerics::FunctorNorm1;
using numerics::FunctorNorm2;
using numerics::FunctorDiagRank;
using numerics::FunctorPrint;

using TensorMethod = talsh::TensorFunctor<Identifiable>;


/** Returns the closest owner id (process rank) for a given subtensor. **/
unsigned int subtensor_owner_id(unsigned int process_rank,          //in: current process rank
                                unsigned int num_processes,         //in: total number of processes
                                unsigned long long subtensor_id,    //in: id of the required subtensor
                                unsigned long long num_subtensors); //in: total number of subtensors

/* Returns a range of subtensors [begin,end] owned by the specified process. */
std::pair<unsigned long long, unsigned long long> owned_subtensors(
                                                   unsigned int process_rank,          //in: target process rank
                                                   unsigned int num_processes,         //in: total number of processes
                                                   unsigned long long num_subtensors); //in: total number of subtensors

/* Returns the subtensor replication level (number of replicated subtensors). */
unsigned int replication_level(const ProcessGroup & process_group, //in: process group
                               std::shared_ptr<Tensor> & tensor);  //in: tensor


//Composite tensor mapper (helper):
class CompositeTensorMapper: public TensorMapper{
public:

#ifdef MPI_ENABLED
 CompositeTensorMapper(const MPICommProxy & communicator,
                       unsigned int current_rank_in_group,
                       unsigned int num_processes_in_group,
                       const std::unordered_map<std::string,std::shared_ptr<Tensor>> & local_tensors):
  current_process_rank_(current_rank_in_group), group_num_processes_(num_processes_in_group),
  intra_comm_(communicator), local_tensors_(local_tensors) {}
#else
 CompositeTensorMapper(unsigned int current_rank_in_group,
                       unsigned int num_processes_in_group,
                       const std::unordered_map<std::string,std::shared_ptr<Tensor>> & local_tensors):
  current_process_rank_(current_rank_in_group), group_num_processes_(num_processes_in_group),
  local_tensors_(local_tensors) {}
#endif

 virtual ~CompositeTensorMapper() = default;

 virtual unsigned int subtensorOwnerId(unsigned long long subtensor_id,
                                       unsigned long long num_subtensors) const override
 {
  return subtensor_owner_id(current_process_rank_,group_num_processes_,subtensor_id,num_subtensors);
 }

 virtual std::pair<unsigned long long, unsigned long long> ownedSubtensors(unsigned int process_rank,
                                                                           unsigned long long num_subtensors) const override
 {
  return owned_subtensors(process_rank,group_num_processes_,num_subtensors);
 }

 virtual bool isLocalSubtensor(unsigned long long subtensor_id,
                               unsigned long long num_subtensors) const override
 {
  return (subtensorOwnerId(subtensor_id,num_subtensors) == current_process_rank_);
 }

 virtual bool isLocalSubtensor(const Tensor & subtensor) const override
 {
  return (local_tensors_.find(subtensor.getName()) != local_tensors_.cend());
 }

 virtual unsigned int getProcessRank() const override {return current_process_rank_;}

 virtual unsigned int getNumProcesses() const override {return group_num_processes_;}

 virtual const MPICommProxy & getMPICommProxy() const override {return intra_comm_;}

private:

 unsigned int current_process_rank_; //rank of the current process (in some process group)
 unsigned int group_num_processes_;  //total number of processes (in some process group)
 MPICommProxy intra_comm_;           //MPI communicator for the process group
 const std::unordered_map<std::string,std::shared_ptr<Tensor>> & local_tensors_; //locally stored tensors
};


//Numerical server:
class NumServer final {

public:

#ifdef MPI_ENABLED
 NumServer(const MPICommProxy & communicator,                               //MPI communicator proxy
           const ParamConf & parameters,                                    //runtime configuration parameters
           const std::string & graph_executor_name = "lazy-dag-executor",   //DAG executor kind
           const std::string & node_executor_name = "talsh-node-executor"); //DAG node executor kind
#else
 NumServer(const ParamConf & parameters,                                    //runtime configuration parameters
           const std::string & graph_executor_name = "lazy-dag-executor",   //DAG executor kind
           const std::string & node_executor_name = "talsh-node-executor"); //DAG node executor kind
#endif

 NumServer(const NumServer &) = delete;
 NumServer & operator=(const NumServer &) = delete;
 NumServer(NumServer &&) noexcept = delete;
 NumServer & operator=(NumServer &&) noexcept = delete;
 ~NumServer();

 /** Reconfigures tensor runtime implementation. **/
#ifdef MPI_ENABLED
 void reconfigureTensorRuntime(const MPICommProxy & communicator,
                               const ParamConf & parameters,
                               const std::string & dag_executor_name,
                               const std::string & node_executor_name);
#else
 void reconfigureTensorRuntime(const ParamConf & parameters,
                               const std::string & dag_executor_name,
                               const std::string & node_executor_name);
#endif

 /** Resets the tensor contraction sequence optimizer that is
     invoked when evaluating tensor networks. **/
 void resetContrSeqOptimizer(const std::string & optimizer_name, //in: tensor contraction sequence optimizer name
                             bool caching = false);              //whether or not optimized tensor contraction sequence will be cached for later reuse

 /** Activates optimized tensor contraction sequence caching for later reuse. **/
 void activateContrSeqCaching(bool persist = false);

 /** Deactivates optimized tensor contraction sequence caching. **/
 void deactivateContrSeqCaching();

 /** Queries the status of optimized tensor contraction sequence caching. **/
 bool queryContrSeqCaching() const;

 /** Resets the client logging level (0:none). **/
 void resetClientLoggingLevel(int level = 0);

 /** Resets the runtime logging level (0:none). **/
 void resetRuntimeLoggingLevel(int level = 0);

 /** Resets tensor operation execution serialization. **/
 void resetExecutionSerialization(bool serialize,
                                  bool validation_trace = false);

 /** Activates/deactivates dry run (no actual computations). **/
 void activateDryRun(bool dry_run);

 /** Activates mixed-precision fast math operations on all devices (if available). **/
 void activateFastMath();

 /** Returns the Host memory buffer size in bytes provided by the runtime. **/
 std::size_t getMemoryBufferSize() const;

 /** Returns the current value of the Flop counter. **/
 double getTotalFlopCount() const;

 /** Returns the default process group comprising all MPI processes and their communicator. **/
 const ProcessGroup & getDefaultProcessGroup() const;

 /** Returns the current process group comprising solely the current MPI process and its own self-communicator. **/
 const ProcessGroup & getCurrentProcessGroup() const;

 /** Returns the local rank of the MPI process in a given process group, or -1 if it does not belong to it. **/
 int getProcessRank(const ProcessGroup & process_group) const;

 /** Returns the global rank of the current MPI process in the default process group. **/
 int getProcessRank() const;

 /** Returns the number of MPI processes in a given process group. **/
 int getNumProcesses(const ProcessGroup & process_group) const;

 /** Returns the total number of MPI processes in the default process group. **/
 int getNumProcesses() const;

 /** Returns a composite tensor mapper for a given process group. **/
 std::shared_ptr<TensorMapper> getTensorMapper(const ProcessGroup & process_group) const;

 /** Returns a composite tensor mapper for the default process group (all processes). **/
 std::shared_ptr<TensorMapper> getTensorMapper() const;

 /** Registers an external tensor method. **/
 void registerTensorMethod(const std::string & tag,
                           std::shared_ptr<TensorMethod> method);

 /** Retrieves a registered external tensor method. **/
 std::shared_ptr<TensorMethod> getTensorMethod(const std::string & tag);

 /** Registers an external data packet. **/
 void registerExternalData(const std::string & tag,
                           std::shared_ptr<BytePacket> packet);

 /** Retrieves a registered external data packet. **/
 std::shared_ptr<BytePacket> getExternalData(const std::string & tag);


 /** Opens a new (child) TAProL scope and returns its id. **/
 ScopeId openScope(const std::string & scope_name); //new scope name

 /** Closes the currently open TAProL scope and returns its parental scope id. **/
 ScopeId closeScope();


 /** Creates a named vector space, returns its registered id, and,
     optionally, a non-owning pointer to it. **/
 SpaceId createVectorSpace(const std::string & space_name,            //in: vector space name
                           DimExtent space_dim,                       //in: vector space dimension
                           const VectorSpace ** space_ptr = nullptr); //out: non-owning pointer to the created vector space

 /** Destroys a previously created named vector space. **/
 void destroyVectorSpace(const std::string & space_name); //in: name of the vector space to destroy
 void destroyVectorSpace(SpaceId space_id);               //in: id of the vector space to destroy

 /** Returns a non-owning pointer to a previosuly registered vector space,
     including the anonymous vector space. **/
 const VectorSpace * getVectorSpace(const std::string & space_name) const;


 /** Creates a named subspace of a named vector space,
     returns its registered id, and, optionally, a non-owning pointer to it. **/
 SubspaceId createSubspace(const std::string & subspace_name,         //in: subspace name
                           const std::string & space_name,            //in: containing vector space name
                           std::pair<DimOffset,DimOffset> bounds,     //in: range of basis vectors defining the created subspace
                           const Subspace ** subspace_ptr = nullptr); //out: non-owning pointer to the created subspace

 /** Destroys a previously created named subspace of a named vector space. **/
 void destroySubspace(const std::string & subspace_name); //in: name of the subspace to destroy
 void destroySubspace(SubspaceId subspace_id); //in: id of the subspace to destroy

 /** Returns a non-owning pointer to a previosuly registered named subspace
     of a previously registered named vector space. **/
 const Subspace * getSubspace(const std::string & subspace_name) const;


 /** Submits an individual (simple or composite) tensor operation for processing.
     Composite tensor operations require an implementation of the TensorMapper interface. **/
 bool submit(std::shared_ptr<TensorOperation> operation,   //in: tensor operation for numerical evaluation
             std::shared_ptr<TensorMapper> tensor_mapper); //in: tensor mapper (for composite tensor operations only)

 /** Submits a tensor network for processing (evaluating the output tensor-result).
     If the output (result) tensor has not been created yet, it will be created and
     initialized to zero automatically, and later destroyed automatically when no longer needed.
     By default all parallel processes will be processing the tensor network,
     otherwise the desired process subset needs to be explicitly specified. **/
 bool submit(TensorNetwork & network);                    //in: tensor network for numerical evaluation
 bool submit(std::shared_ptr<TensorNetwork> network);     //in: tensor network for numerical evaluation
 bool submit(const ProcessGroup & process_group,          //in: chosen group of MPI processes
             TensorNetwork & network);                    //in: tensor network for numerical evaluation
 bool submit(const ProcessGroup & process_group,          //in: chosen group of MPI processes
             std::shared_ptr<TensorNetwork> network);     //in: tensor network for numerical evaluation

 /** Submits a tensor network expansion for processing (evaluating output tensors of all
     constituting tensor networks and accumualting them in the provided accumulator tensor).
     Synchronization of the tensor expansion evaluation is done via syncing on the accumulator
     tensor. By default all parallel processes will be processing the tensor network,
     otherwise the desired process subset needs to be explicitly specified. **/
 bool submit(TensorExpansion & expansion,                 //in: tensor expansion for numerical evaluation
             std::shared_ptr<Tensor> accumulator);        //inout: tensor accumulator (result)
 bool submit(std::shared_ptr<TensorExpansion> expansion,  //in: tensor expansion for numerical evaluation
             std::shared_ptr<Tensor> accumulator);        //inout: tensor accumulator (result)
 bool submit(const ProcessGroup & process_group,          //in: chosen group of MPI processes
             TensorExpansion & expansion,                 //in: tensor expansion for numerical evaluation
             std::shared_ptr<Tensor> accumulator);        //inout: tensor accumulator (result)
 bool submit(const ProcessGroup & process_group,          //in: chosen group of MPI processes
             std::shared_ptr<TensorExpansion> expansion,  //in: tensor expansion for numerical evaluation
             std::shared_ptr<Tensor> accumulator);        //inout: tensor accumulator (result)

 /** Synchronizes all update operations on a given tensor.
     Changing wait to FALSE, only tests for completion.
     If ProcessGroup is not provided, defaults to the local process. **/
 bool sync(const Tensor & tensor,
           bool wait = true);
 bool sync(const ProcessGroup & process_group,
           const Tensor & tensor,
           bool wait = true);
 /** Synchronizes execution of a specific tensor operation.
     Changing wait to FALSE will only test for completion.
    `This method has local synchronization semantics! **/
 bool sync(TensorOperation & operation,
           bool wait = true);
 /** Synchronizes execution of a specific tensor network.
     Changing wait to FALSE, only tests for completion.
     If ProcessGroup is not provided, defaults to the local process. **/
 bool sync(TensorNetwork & network,
           bool wait = true);
 bool sync(const ProcessGroup & process_group,
           TensorNetwork & network,
           bool wait = true);
 /** Synchronizes execution of all outstanding tensor operations.
     Changing wait to FALSE, only tests for completion.
     If ProcessGroup is not provided, defaults to the local process. **/
 bool sync(bool wait = true);
 bool sync(const ProcessGroup & process_group,
           bool wait = true);

 /** HIGHER-LEVEL WRAPPERS **/

 /** Synchronizes all outstanding update operations on a given tensor.
     If ProcessGroup is not provided, defaults to the local process. **/
 bool sync(const std::string & name, //in: tensor name
           bool wait = true);        //in: wait versus test for completion
 bool sync(const ProcessGroup & process_group,
           const std::string & name, //in: tensor name
           bool wait = true);        //in: wait versus test for completion

 /** Checks whether a given tensor has been allocated storage (created). **/
 bool tensorAllocated(const std::string & name) const; //in: tensor name

 /** Returns a shared pointer to the requested tensor object. **/
 std::shared_ptr<Tensor> getTensor(const std::string & name); //in: tensor name

 /** Returns a reference to the requested tensor object. **/
 Tensor & getTensorRef(const std::string & name); //in: tensor name

 /** Returns the tensor element type. **/
 TensorElementType getTensorElementType(const std::string & name) const; //in: tensor name

 /** Registers a group of tensor dimensions which form an isometry when
     contracted over with the conjugated tensor (see exatn::numerics::Tensor).
     Returns TRUE on success, FALSE on failure. **/
 bool registerTensorIsometry(const std::string & name,                     //in: tensor name
                             const std::vector<unsigned int> & iso_dims);  //in: tensor dimensions forming the isometry
 bool registerTensorIsometry(const std::string & name,                     //in: tensor name
                             const std::vector<unsigned int> & iso_dims0,  //in: tensor dimensions forming the isometry (group 0)
                             const std::vector<unsigned int> & iso_dims1); //in: tensor dimensions forming the isometry (group 1)

 /** Returns TRUE if the calling process is within the existence domain of all given tensors, FALSE otherwise. **/
 template <typename... Args>
 bool withinTensorExistenceDomain(const std::string & tensor_name, Args&&... tensor_names) const //in: tensor names
 {
  if(!withinTensorExistenceDomain(tensor_name)) return false;
  return withinTensorExistenceDomain(std::forward<Args>(tensor_names)...);
 }

 bool withinTensorExistenceDomain(const std::string & tensor_name) const; //in: tensor name

 /** Returns the process group associated with the given tensors.
     The calling process must be within the tensor exsistence domain,
     which must be the same for all tensors. **/
 template <typename... Args>
 const ProcessGroup & getTensorProcessGroup(const std::string & tensor_name, Args&&... tensor_names) const //in: tensor names
 {
  const auto & tensor_domain = getTensorProcessGroup(tensor_name);
  const auto & other_tensors_domain = getTensorProcessGroup(std::forward<Args>(tensor_names)...);
  assert(other_tensors_domain == tensor_domain);
  return tensor_domain;
 }

 const ProcessGroup & getTensorProcessGroup(const std::string & tensor_name) const; //tensor name

 /** Declares, registers, and actually creates a tensor via the processing backend.
     See numerics::Tensor constructors for different creation options. **/
 template <typename... Args>
 bool createTensor(const std::string & name,       //in: tensor name
                   TensorElementType element_type, //in: tensor element type
                   Args&&... args);                //in: other arguments for Tensor ctor

 template <typename... Args>
 bool createTensorSync(const std::string & name,       //in: tensor name
                       TensorElementType element_type, //in: tensor element type
                       Args&&... args);                //in: other arguments for Tensor ctor

 bool createTensor(std::shared_ptr<Tensor> tensor,  //in: existing declared tensor (can be composite)
                   TensorElementType element_type); //in: tensor element type

 bool createTensorSync(std::shared_ptr<Tensor> tensor,  //in: existing declared tensor (can be composite)
                       TensorElementType element_type); //in: tensor element type

 bool createTensor(const std::string & name,          //in: tensor name
                   const TensorSignature & signature, //in: tensor signature with registered spaces/subspaces
                   TensorElementType element_type);   //in: tensor element type

 template <typename... Args>
 bool createTensor(const ProcessGroup & process_group, //in: chosen group of MPI processes
                   const std::string & name,           //in: tensor name
                   TensorElementType element_type,     //in: tensor element type
                   Args&&... args);                    //in: other arguments for Tensor ctor

 template <typename... Args>
 bool createTensorSync(const ProcessGroup & process_group, //in: chosen group of MPI processes
                       const std::string & name,           //in: tensor name
                       TensorElementType element_type,     //in: tensor element type
                       Args&&... args);                    //in: other arguments for Tensor ctor

 bool createTensor(const ProcessGroup & process_group, //in: chosen group of MPI processes
                   std::shared_ptr<Tensor> tensor,     //in: existing declared tensor (can be composite)
                   TensorElementType element_type);    //in: tensor element type

 bool createTensorSync(const ProcessGroup & process_group, //in: chosen group of MPI processes
                       std::shared_ptr<Tensor> tensor,     //in: existing declared tensor (can be composite)
                       TensorElementType element_type);    //in: tensor element type

 /** Creates and allocates storage for a composite tensor distributed over a given process group.
     The distribution of tensor blocks is dictated by its split dimensions. **/
 template <typename... Args>
 bool createTensor(const ProcessGroup & process_group,                      //in: chosen group of MPI processes
                   const std::string & name,                                //in: tensor name
                   const std::vector<std::pair<unsigned int,
                                               unsigned int>> & split_dims, //in: split tensor dimensions: pair{Dimension,MaxDepth}
                   TensorElementType element_type,                          //in: tensor element type
                   Args&&... args);                                         //in: other arguments for Tensor ctor

 template <typename... Args>
 bool createTensorSync(const ProcessGroup & process_group,                      //in: chosen group of MPI processes
                       const std::string & name,                                //in: tensor name
                       const std::vector<std::pair<unsigned int,
                                                   unsigned int>> & split_dims, //in: split tensor dimensions: pair{Dimension,MaxDepth}
                       TensorElementType element_type,                          //in: tensor element type
                       Args&&... args);                                         //in: other arguments for Tensor ctor

 /** Creates all tensors in a given tensor network that are still unallocated. **/
 bool createTensors(TensorNetwork & tensor_network,         //inout: tensor network
                    TensorElementType element_type);        //in: tensor element type

 bool createTensorsSync(TensorNetwork & tensor_network,     //inout: tensor network
                        TensorElementType element_type);    //in: tensor element type

 bool createTensors(const ProcessGroup & process_group,     //in: chosen group of MPI processes
                    TensorNetwork & tensor_network,         //inout: tensor network
                    TensorElementType element_type);        //in: tensor element type

 bool createTensorsSync(const ProcessGroup & process_group, //in: chosen group of MPI processes
                        TensorNetwork & tensor_network,     //inout: tensor network
                        TensorElementType element_type);    //in: tensor element type

 /** Creates all tensors in a given tensor network expansion that are still unallocated. **/
 bool createTensors(TensorExpansion & tensor_expansion,     //inout: tensor expansion
                    TensorElementType element_type);        //in: tensor element type

 bool createTensorsSync(TensorExpansion & tensor_expansion, //inout: tensor expansion
                        TensorElementType element_type);    //in: tensor element type

 bool createTensors(const ProcessGroup & process_group,     //in: chosen group of MPI processes
                    TensorExpansion & tensor_expansion,     //inout: tensor expansion
                    TensorElementType element_type);        //in: tensor element type

 bool createTensorsSync(const ProcessGroup & process_group, //in: chosen group of MPI processes
                        TensorExpansion & tensor_expansion, //inout: tensor expansion
                        TensorElementType element_type);    //in: tensor element type

 /** Destroys a tensor, including its backend representation. **/
 bool destroyTensor(const std::string & name); //in: tensor name

 bool destroyTensorSync(const std::string & name); //in: tensor name

 /** Destroys all currently allocated tensors in a given tensor network.
     Note that the destroyed tensors could also be present in other tensor networks. **/
 bool destroyTensors(TensorNetwork & tensor_network);     //inout: tensor network

 bool destroyTensorsSync(TensorNetwork & tensor_network); //inout: tensor network

 /** Initializes a tensor to some scalar value. **/
 template<typename NumericType>
 bool initTensor(const std::string & name, //in: tensor name
                 NumericType value);       //in: scalar value

 template<typename NumericType>
 bool initTensorSync(const std::string & name, //in: tensor name
                     NumericType value);       //in: scalar value

 /** Initializes a tensor with externally provided data.
     The vector containing externally provided data assumes
     the column-wise storage in the initialized tensor.  **/
 template<typename NumericType>
 bool initTensorData(const std::string & name,                   //in: tensor name
                     const std::vector<NumericType> & ext_data); //in: vector with externally provided data

 template<typename NumericType>
 bool initTensorDataSync(const std::string & name,                   //in: tensor name
                         const std::vector<NumericType> & ext_data); //in: vector with externally provided data

/** Initializes a tensor with externally provided data read from a file with format:
     Storage format (string: {dense|list})
     Tensor name
     Tensor shape (space-separated dimension extents)
     Tensor signature (space-separated dimension base offsets)
     Tensor elements:
      Dense format: Numeric values (column-wise order), any number of values per line
      List format: Numeric value and Multi-index in each line **/
 bool initTensorFile(const std::string & name,      //in: tensor name
                     const std::string & filename); //in: file name with tensor data

 bool initTensorFileSync(const std::string & name,      //in: tensor name
                         const std::string & filename); //in: file name with tensor data

 /** Initializes a tensor to some random value. **/
 bool initTensorRnd(const std::string & name);     //in: tensor name

 bool initTensorRndSync(const std::string & name); //in: tensor name

 /** Initializes all input tensors of a given tensor network to a random value. **/
 bool initTensorsRnd(TensorNetwork & tensor_network);     //inout: tensor network

 bool initTensorsRndSync(TensorNetwork & tensor_network); //inout: tensor network

 /** Computes max-abs norm of a tensor. **/
 bool computeMaxAbsSync(const std::string & name, //in: tensor name
                        double & norm);           //out: tensor norm

 /** Computes 1-norm of a tensor. **/
 bool computeNorm1Sync(const std::string & name, //in: tensor name
                       double & norm);           //out: tensor norm

 /** Computes 2-norm of a tensor. **/
 bool computeNorm2Sync(const std::string & name, //in: tensor name
                       double & norm);           //out: tensor norm

 /** Computes partial 2-norms over a chosen tensor dimension. **/
 bool computePartialNormsSync(const std::string & name,             //in: tensor name
                              unsigned int tensor_dimension,        //in: chosen tensor dimension
                              std::vector<double> & partial_norms); //out: partial 2-norms over the chosen tensor dimension

 /** Replicates a tensor within the given process group, which defaults to all MPI processes.
     Only the root_process_rank within the given process group is required to have the tensor,
     that is, the tensor will automatically be created in those MPI processes which do not have it.  **/
 bool replicateTensor(const std::string & name,           //in: tensor name
                      int root_process_rank);             //in: local rank of the root process within the given process group

 bool replicateTensorSync(const std::string & name,       //in: tensor name
                          int root_process_rank);         //in: local rank of the root process within the given process group

 bool replicateTensor(const ProcessGroup & process_group, //in: chosen group of MPI processes
                      const std::string & name,           //in: tensor name
                      int root_process_rank);             //in: local rank of the root process within the given process group

 bool replicateTensorSync(const ProcessGroup & process_group, //in: chosen group of MPI processes
                          const std::string & name,           //in: tensor name
                          int root_process_rank);             //in: local rank of the root process within the given process group

 /** Broadcast a tensor among all MPI processes within a given process group,
     which defaults to all MPI processes. This function is needed when
     a tensor is updated in an operation submitted to a subset of MPI processes
     such that the excluded MPI processes can receive an updated version of the tensor.
     Note that the tensor must exist in all participating MPI processes. **/
 bool broadcastTensor(const std::string & name,           //in: tensor name
                      int root_process_rank);             //in: local rank of the root process within the given process group

 bool broadcastTensorSync(const std::string & name,       //in: tensor name
                          int root_process_rank);         //in: local rank of the root process within the given process group

 bool broadcastTensor(const ProcessGroup & process_group, //in: chosen group of MPI processes
                      const std::string & name,           //in: tensor name
                      int root_process_rank);             //in: local rank of the root process within the given process group

 bool broadcastTensorSync(const ProcessGroup & process_group, //in: chosen group of MPI processes
                          const std::string & name,           //in: tensor name
                          int root_process_rank);             //in: local rank of the root process within the given process group

 /** Performs a global sum reduction on a tensor among all MPI processes within a given
     process group, which defaults to all MPI processes. This function is needed when
     multiple MPI processes compute their local updates to the tensor, thus requiring
     a global sum reduction such that each MPI process will get the final (same) tensor
     value. Note that the tensor must exist in all participating MPI processes. **/
 bool allreduceTensor(const std::string & name);          //in: tensor name

 bool allreduceTensorSync(const std::string & name);      //in: tensor name

 bool allreduceTensor(const ProcessGroup & process_group, //in: chosen group of MPI processes
                      const std::string & name);          //in: tensor name

 bool allreduceTensorSync(const ProcessGroup & process_group, //in: chosen group of MPI processes
                          const std::string & name);          //in: tensor name

 /** Scales a tensor by a scalar value. **/
 template<typename NumericType>
 bool scaleTensor(const std::string & name, //in: tensor name
                  NumericType value);       //in: scalar value

 template<typename NumericType>
 bool scaleTensorSync(const std::string & name, //in: tensor name
                      NumericType value);       //in: scalar value

 /** Transforms (updates) a tensor according to a user-defined tensor functor. **/
 bool transformTensor(const std::string & name,               //in: tensor name
                      std::shared_ptr<TensorMethod> functor); //in: functor defining the tensor transformation

 bool transformTensorSync(const std::string & name,               //in: tensor name
                          std::shared_ptr<TensorMethod> functor); //in: functor defining the tensor transformation

 bool transformTensor(const std::string & name,              //in: tensor name
                      const std::string & functor_name);     //in: name of the functor defining the tensor transformation

 bool transformTensorSync(const std::string & name,          //in: tensor name
                          const std::string & functor_name); //in: name of the functor defining the tensor transformation

 /** Extracts a slice from a tensor and stores it in another tensor
     the signature and shape of which determines which slice to extract. **/
 bool extractTensorSlice(const std::string & tensor_name, //in: tensor name
                         const std::string & slice_name); //in: slice name

 bool extractTensorSliceSync(const std::string & tensor_name, //in: tensor name
                             const std::string & slice_name); //in: slice name

 /** Inserts a slice into a tensor. The signature and shape of the slice
     determines the position in the tensor where the slice will be inserted. **/
 bool insertTensorSlice(const std::string & tensor_name, //in: tensor name
                        const std::string & slice_name); //in: slice name

 bool insertTensorSliceSync(const std::string & tensor_name, //in: tensor name
                            const std::string & slice_name); //in: slice name

 /** Assigns one tensor to another congruent one (makes a copy of a tensor).
     If the output tensor with the given name does not exist, it will be created.
     Note that the output tensor must either exist or not exist across all
     participating processes, otherwise it will result in an undefined behavior! **/
 bool copyTensor(const std::string & output_name, //in: output tensor name
                 const std::string & input_name); //in: input tensor name

 bool copyTensorSync(const std::string & output_name, //in: output tensor name
                     const std::string & input_name); //in: input tensor name

 /** Performs tensor addition: tensor0 += tensor1 * alpha **/
 template<typename NumericType>
 bool addTensors(const std::string & addition, //in: symbolic tensor addition specification
                 NumericType alpha);           //in: alpha prefactor

 template<typename NumericType>
 bool addTensorsSync(const std::string & addition, //in: symbolic tensor addition specification
                     NumericType alpha);           //in: alpha prefactor

 /** Performs tensor contraction: tensor0 += tensor1 * tensor2 * alpha **/
 template<typename NumericType>
 bool contractTensors(const std::string & contraction, //in: symbolic tensor contraction specification
                      NumericType alpha);              //in: alpha prefactor

 template<typename NumericType>
 bool contractTensorsSync(const std::string & contraction, //in: symbolic tensor contraction specification
                          NumericType alpha);              //in: alpha prefactor

 /** Decomposes a tensor into three tensor factors via SVD. The symbolic
     tensor contraction specification specifies the decomposition,
     for example:
      D(a,b,c,d,e) = L(c,i,e,j) * S(i,j) * R(b,j,a,i,d)
     where
      L(c,i,e,j) is the left SVD factor,
      R(b,j,a,i,d) is the right SVD factor,
      S(i,j) is the middle SVD factor (the diagonal with singular values).
     Note that the ordering of the contracted indices is not guaranteed! **/
 bool decomposeTensorSVD(const std::string & contraction); //in: three-factor symbolic tensor contraction specification

 bool decomposeTensorSVDSync(const std::string & contraction); //in: three-factor symbolic tensor contraction specification

 /** Decomposes a tensor into two tensor factors via SVD. The symbolic
     tensor contraction specification specifies the decomposition,
     for example:
      D(a,b,c,d,e) = L(c,i,e,j) * R(b,j,a,i,d)
     where
      L(c,i,e,j) is the left SVD factor with absorbed singular values,
      R(b,j,a,i,d) is the right SVD factor. **/
 bool decomposeTensorSVDL(const std::string & contraction); //in: two-factor symbolic tensor contraction specification

 bool decomposeTensorSVDLSync(const std::string & contraction); //in: two-factor symbolic tensor contraction specification

 /** Decomposes a tensor into two tensor factors via SVD. The symbolic
     tensor contraction specification specifies the decomposition,
     for example:
      D(a,b,c,d,e) = L(c,i,e,j) * R(b,j,a,i,d)
     where
      L(c,i,e,j) is the left SVD factor,
      R(b,j,a,i,d) is the right SVD factor with absorbed singular values. **/
 bool decomposeTensorSVDR(const std::string & contraction); //in: two-factor symbolic tensor contraction specification

 bool decomposeTensorSVDRSync(const std::string & contraction); //in: two-factor symbolic tensor contraction specification

 /** Decomposes a tensor into two tensor factors via SVD. The symbolic
     tensor contraction specification specifies the decomposition,
     for example:
      D(a,b,c,d,e) = L(c,i,e,j) * R(b,j,a,i,d)
     where
      L(c,i,e,j) is the left SVD factor with absorbed square root of singular values,
      R(b,j,a,i,d) is the right SVD factor with absorbed square root of singular values. **/
 bool decomposeTensorSVDLR(const std::string & contraction); //in: two-factor symbolic tensor contraction specification

 bool decomposeTensorSVDLRSync(const std::string & contraction); //in: two-factor symbolic tensor contraction specification

 /** Orthogonalizes a tensor by decomposing it via SVD while discarding
     the middle tensor factor with singular values. The symbolic tensor contraction
     specification specifies the decomposition. It must contain strictly one contracted index! **/
 bool orthogonalizeTensorSVD(const std::string & contraction); //in: two-factor symbolic tensor contraction specification

 bool orthogonalizeTensorSVDSync(const std::string & contraction); //in: two-factor symbolic tensor contraction specification

 /** Orthogonalizes an isometric tensor over its isometric dimensions via the modified Gram-Schmidt procedure.  **/
 bool orthogonalizeTensorMGS(const std::string & name); //in: tensor name

 bool orthogonalizeTensorMGSSync(const std::string & name); //in: tensor name

 /** Prints a tensor to the standard output. **/
 bool printTensor(const std::string & name); //in: tensor name

 bool printTensorSync(const std::string & name); //in: tensor name

 /** Prints a tensor to a file. **/
 bool printTensorFile(const std::string & name,      //in: tensor name
                      const std::string & filename); //in: file name

 bool printTensorFileSync(const std::string & name,      //in: tensor name
                          const std::string & filename); //in: file name

 /** Performs a full evaluation of a tensor network based on the symbolic
     specification involving already created tensors (including the output). **/
 bool evaluateTensorNetwork(const std::string & name,           //in: tensor network name
                            const std::string & network);       //in: symbolic tensor network specification
 bool evaluateTensorNetwork(const ProcessGroup & process_group, //in: chosen group of MPI processes
                            const std::string & name,           //in: tensor network name
                            const std::string & network);       //in: symbolic tensor network specification

 bool evaluateTensorNetworkSync(const std::string & name,           //in: tensor network name
                                const std::string & network);       //in: symbolic tensor network specification
 bool evaluateTensorNetworkSync(const ProcessGroup & process_group, //in: chosen group of MPI processes
                                const std::string & name,           //in: tensor network name
                                const std::string & network);       //in: symbolic tensor network specification

 /** Normalizes a tensor to a given 2-norm. **/
 bool normalizeNorm2Sync(const std::string & name, //in: tensor name
                         double norm);             //in: desired 2-norm

 /** Normalizes a tensor network expansion to a given 2-norm by rescaling
     all tensor network components by the same factor: Only the tensor
     network expansion coefficients are affected. **/
 bool normalizeNorm2Sync(TensorExpansion & expansion, //inout: tensor network expansion
                         double norm);                //in: desired 2-norm

 bool normalizeNorm2Sync(const ProcessGroup & process_group, //in: chosen group of MPI processes
                         TensorExpansion & expansion,        //inout: tensor network expansion
                         double norm);                       //in: desired 2-norm

 /** Normalizes all input tensors in a tensor network to a given 2-norm.
     If only_optimizable is TRUE, only optimizable tensors will be normalized. **/
 bool balanceNorm2Sync(TensorNetwork & network,        //inout: tensor network
                       double norm,                    //in: desired 2-norm
                       bool only_optimizable = false); //in: whether to normalize only optimizable tensors

 bool balanceNorm2Sync(const ProcessGroup & process_group, //in: chosen group of MPI processes
                       TensorNetwork & network,            //inout: tensor network
                       double norm,                        //in: desired 2-norm
                       bool only_optimizable = false);     //in: whether to normalize only optimizable tensors

 /** Normalizes all input tensors in a tensor network expansion to a given 2-norm.
     If only_optimizable is TRUE, only optimizable tensors will be normalized. **/
 bool balanceNorm2Sync(TensorExpansion & expansion,    //inout: tensor network expansion
                       double norm,                    //in: desired 2-norm
                       bool only_optimizable = false); //in: whether to normalize only optimizable tensors

 bool balanceNorm2Sync(const ProcessGroup & process_group, //in: chosen group of MPI processes
                       TensorExpansion & expansion,        //inout: tensor network expansion
                       double norm,                        //in: desired 2-norm
                       bool only_optimizable = false);     //in: whether to normalize only optimizable tensors

 /** Normalizes all input tensors in a tensor network expansion to a given 2-norm
     and rescales tensor network expansion coefficients to normalize the entire
     tensor network expansion to another given 2-norm. If only_optimizable is TRUE,
     only optimizable tensors will be normalized. **/
 bool balanceNormalizeNorm2Sync(TensorExpansion & expansion,    //inout: tensor network expansion
                                double tensor_norm = 1.0,       //in: desired 2-norm of each input tensor
                                double expansion_norm = 1.0,    //in: desired 2-norm of the tensor network expansion
                                bool only_optimizable = false); //in: whether to normalize only optimizable tensors

 bool balanceNormalizeNorm2Sync(const ProcessGroup & process_group, //in: chosen group of MPI processes
                                TensorExpansion & expansion,        //inout: tensor network expansion
                                double tensor_norm = 1.0,           //in: desired 2-norm of each input tensor
                                double expansion_norm = 1.0,        //in: desired 2-norm of the tensor network expansion
                                bool only_optimizable = false);     //in: whether to normalize only optimizable tensors

 /** Returns a locally stored tensor slice (talsh::Tensor) providing access to tensor elements.
     This slice will be extracted from the exatn::numerics::Tensor implementation as a copy.
     The returned future becomes ready once the execution thread has retrieved the slice copy. **/
 std::shared_ptr<talsh::Tensor> getLocalTensor(std::shared_ptr<Tensor> tensor, //in: exatn::numerics::Tensor to get slice of (by copy)
              const std::vector<std::pair<DimOffset,DimExtent>> & slice_spec); //in: tensor slice specification
 /** This overload will return a copy of the full tensor. **/
 std::shared_ptr<talsh::Tensor> getLocalTensor(std::shared_ptr<Tensor> tensor);
 /** This overload references the ExaTN tensor by its registered name. **/
 std::shared_ptr<talsh::Tensor> getLocalTensor(const std::string & name, //in: exatn tensor name
        const std::vector<std::pair<DimOffset,DimExtent>> & slice_spec); //in: tensor slice specification
 /** This overload returns a copy of the full tensor while referencing it by its registered name. **/
 std::shared_ptr<talsh::Tensor> getLocalTensor(const std::string & name); //in: exatn tensor name

 inline double getTimeStampStart() const {return time_start_;}

 /** DEBUG: Prints all currently existing tensors created implicitly. **/
 void printImplicitTensors() const;

protected:

 /** Submits an individual tensor operation for processing. **/
 bool submitOp(std::shared_ptr<TensorOperation> operation); //in: tensor operation for numerical evaluation

 /** Destroys orphaned tensors (garbage collection). **/
 void destroyOrphanedTensors();

private:

 std::shared_ptr<numerics::SpaceRegister> space_register_; //register of vector spaces and their named subspaces
 std::unordered_map<std::string,SpaceId> subname2id_; //maps a subspace name to its parental vector space id

 std::unordered_map<std::string,std::shared_ptr<Tensor>> tensors_; //registered tensors (by CREATE operation)
 std::list<std::shared_ptr<Tensor>> implicit_tensors_; //tensors created implicitly by the runtime (for garbage collection)
 std::unordered_map<std::string,ProcessGroup> tensor_comms_; //process group associated with each tensor

 std::string contr_seq_optimizer_; //tensor contraction sequence optimizer invoked when evaluating tensor networks
 bool contr_seq_caching_; //regulates whether or not to cache pseudo-optimal tensor contraction orders for later reuse

 std::map<std::string,std::shared_ptr<TensorMethod>> ext_methods_; //external tensor methods
 std::map<std::string,std::shared_ptr<BytePacket>> ext_data_; //external data

 std::stack<std::pair<std::string,ScopeId>> scopes_; //TAProL scope stack: {Scope name, Scope Id}

 TensorOpFactory * tensor_op_factory_; //tensor operation factory (non-owning pointer)

 int logging_; //logging level
 std::ofstream logfile_; //log file
 int num_processes_; //total number of parallel processes in the dedicated MPI communicator
 int process_rank_; //rank of the current parallel process in the dedicated MPI communicator
 int global_process_rank_; //rank of the current parallel process in MPI_COMM_WORLD
 MPICommProxy intra_comm_; //dedicated MPI intra-communicator used to initialize the Numerical Server
 std::shared_ptr<TensorMapper> default_tensor_mapper_; //default composite tensor mapper (across all parallel processes)
 std::shared_ptr<ProcessGroup> process_world_; //default process group comprising all MPI processes and their communicator
 std::shared_ptr<ProcessGroup> process_self_;  //current process group comprising solely the current MPI process and its own communicator
 std::shared_ptr<runtime::TensorRuntime> tensor_rt_; //tensor runtime (for actual execution of tensor operations)
 BytePacket byte_packet_; //byte packet for exchanging tensor meta-data
 double time_start_; //time stamp of the Numerical Server start
 bool validation_tracing_; //validation tracing flag (for debugging)
};

/** Numerical service singleton (numerical server) **/
extern std::shared_ptr<NumServer> numericalServer;


//TEMPLATE DEFINITIONS:
template <typename... Args>
bool NumServer::createTensor(const std::string & name,
                             TensorElementType element_type,
                             Args&&... args)
{
 return createTensor(getDefaultProcessGroup(),name,element_type,std::forward<Args>(args)...);
}

template <typename... Args>
bool NumServer::createTensorSync(const std::string & name,
                                 TensorElementType element_type,
                                 Args&&... args)
{
 return createTensorSync(getDefaultProcessGroup(),name,element_type,std::forward<Args>(args)...);
}

template <typename... Args>
bool NumServer::createTensor(const ProcessGroup & process_group,
                             const std::string & name,
                             TensorElementType element_type,
                             Args&&... args)
{
 if(!process_group.rankIsIn(process_rank_)) return true; //process is not in the group: Do nothing
 bool submitted = false;
 if(element_type != TensorElementType::VOID){
  std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::CREATE);
  op->setTensorOperand(std::make_shared<Tensor>(name,std::forward<Args>(args)...));
  std::dynamic_pointer_cast<numerics::TensorOpCreate>(op)->resetTensorElementType(element_type);
  submitted = submit(op,getTensorMapper(process_group));
  if(submitted){
   if(process_group != getDefaultProcessGroup()){
    auto saved = tensor_comms_.emplace(std::make_pair(name,process_group));
    assert(saved.second);
   }
  }
 }else{
  std::cout << "#ERROR(exatn::createTensor): Missing data type!" << std::endl;
 }
 return submitted;
}

template <typename... Args>
bool NumServer::createTensorSync(const ProcessGroup & process_group,
                                 const std::string & name,
                                 TensorElementType element_type,
                                 Args&&... args)
{
 if(!process_group.rankIsIn(process_rank_)) return true; //process is not in the group: Do nothing
 bool submitted = false;
 if(element_type != TensorElementType::VOID){
  std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::CREATE);
  op->setTensorOperand(std::make_shared<Tensor>(name,std::forward<Args>(args)...));
  std::dynamic_pointer_cast<numerics::TensorOpCreate>(op)->resetTensorElementType(element_type);
  submitted = submit(op,getTensorMapper(process_group));
  if(submitted){
   if(process_group != getDefaultProcessGroup()){
    auto saved = tensor_comms_.emplace(std::make_pair(name,process_group));
    assert(saved.second);
   }
   submitted = sync(*op);
  }
 }else{
  std::cout << "#ERROR(exatn::createTensor): Missing data type!" << std::endl;
 }
 return submitted;
}

template <typename... Args>
bool NumServer::createTensor(const ProcessGroup & process_group,                      //in: chosen group of MPI processes
                             const std::string & name,                                //in: tensor name
                             const std::vector<std::pair<unsigned int,
                                                         unsigned int>> & split_dims, //in: split tensor dimensions: pair{Dimension,MaxDepth}
                             TensorElementType element_type,                          //in: tensor element type
                             Args&&... args)                                          //in: other arguments for Tensor ctor
{
 return createTensor(process_group,makeSharedTensorComposite(split_dims,name,std::forward<Args>(args)...),element_type);
}

template <typename... Args>
bool NumServer::createTensorSync(const ProcessGroup & process_group,                      //in: chosen group of MPI processes
                                 const std::string & name,                                //in: tensor name
                                 const std::vector<std::pair<unsigned int,
                                                             unsigned int>> & split_dims, //in: split tensor dimensions: pair{Dimension,MaxDepth}
                                 TensorElementType element_type,                          //in: tensor element type
                                 Args&&... args)                                          //in: other arguments for Tensor ctor
{
 return createTensorSync(process_group,makeSharedTensorComposite(split_dims,name,std::forward<Args>(args)...),element_type);
}

template<typename NumericType>
bool NumServer::initTensor(const std::string & name,
                           NumericType value)
{
 return transformTensor(name,std::shared_ptr<TensorMethod>(new numerics::FunctorInitVal(value)));
}

template<typename NumericType>
bool NumServer::initTensorSync(const std::string & name,
                               NumericType value)
{
 return transformTensorSync(name,std::shared_ptr<TensorMethod>(new numerics::FunctorInitVal(value)));
}

template<typename NumericType>
bool NumServer::initTensorData(const std::string & name,
                               const std::vector<NumericType> & ext_data)
{
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()) return false;
 return transformTensor(name,std::shared_ptr<TensorMethod>(
         new numerics::FunctorInitDat(iter->second->getShape(),ext_data)));
}

template<typename NumericType>
bool NumServer::initTensorDataSync(const std::string & name,
                                   const std::vector<NumericType> & ext_data)
{
 auto iter = tensors_.find(name);
 if(iter == tensors_.end()) return false;
 return transformTensorSync(name,std::shared_ptr<TensorMethod>(
         new numerics::FunctorInitDat(iter->second->getShape(),ext_data)));
}

template<typename NumericType>
bool NumServer::scaleTensor(const std::string & name,
                            NumericType value)
{
 return transformTensor(name,std::shared_ptr<TensorMethod>(new numerics::FunctorScale(value)));
}

template<typename NumericType>
bool NumServer::scaleTensorSync(const std::string & name,
                                NumericType value)
{
 return transformTensorSync(name,std::shared_ptr<TensorMethod>(new numerics::FunctorScale(value)));
}

template<typename NumericType>
bool NumServer::addTensors(const std::string & addition,
                           NumericType alpha)
{
 std::vector<std::string> tensors;
 auto parsed = parse_tensor_network(addition,tensors);
 if(parsed){
  if(tensors.size() == 2){
   std::string tensor_name;
   std::vector<IndexLabel> indices;
   bool complex_conj0,complex_conj1;
   parsed = parse_tensor(tensors[0],tensor_name,indices,complex_conj0);
   if(parsed){
    assert(!complex_conj0);
    auto iter = tensors_.find(tensor_name);
    if(iter != tensors_.end()){
     auto tensor0 = iter->second;
     parsed = parse_tensor(tensors[1],tensor_name,indices,complex_conj1);
     if(parsed){
      iter = tensors_.find(tensor_name);
      if(iter != tensors_.end()){
       auto tensor1 = iter->second;
       const auto & process_group = getTensorProcessGroup(tensor0->getName(),tensor1->getName());
       std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::ADD);
       op->setTensorOperand(tensor0,complex_conj0);
       op->setTensorOperand(tensor1,complex_conj1);
       op->setIndexPattern(addition);
       op->setScalar(0,std::complex<double>(alpha));
       parsed = submit(op,getTensorMapper(process_group));
      }else{
       parsed = true;
       //std::cout << "#ERROR(exatn::NumServer::addTensors): Tensor " << tensor_name << " not found in tensor addition: "
       //          << addition << std::endl;
      }
     }else{
      std::cout << "#ERROR(exatn::NumServer::addTensors): Invalid argument#1 in tensor addition: "
                << addition << std::endl;
     }
    }else{
     parsed = true;
     //std::cout << "#ERROR(exatn::NumServer::addTensors): Tensor " << tensor_name << " not found in tensor addition: "
     //          << addition << std::endl;
    }
   }else{
    std::cout << "#ERROR(exatn::NumServer::addTensors): Invalid argument#0 in tensor addition: "
              << addition << std::endl;
   }
  }else{
   parsed = false;
   std::cout << "#ERROR(exatn::NumServer::addTensors): Invalid number of arguments in tensor addition: "
             << addition << std::endl;
  }
 }else{
  std::cout << "#ERROR(exatn::NumServer::addTensors): Invalid tensor addition: " << addition << std::endl;
 }
 return parsed;
}

template<typename NumericType>
bool NumServer::addTensorsSync(const std::string & addition,
                               NumericType alpha)
{
 std::vector<std::string> tensors;
 auto parsed = parse_tensor_network(addition,tensors);
 if(parsed){
  if(tensors.size() == 2){
   std::string tensor_name;
   std::vector<IndexLabel> indices;
   bool complex_conj0,complex_conj1;
   parsed = parse_tensor(tensors[0],tensor_name,indices,complex_conj0);
   if(parsed){
    assert(!complex_conj0);
    auto iter = tensors_.find(tensor_name);
    if(iter != tensors_.end()){
     auto tensor0 = iter->second;
     parsed = parse_tensor(tensors[1],tensor_name,indices,complex_conj1);
     if(parsed){
      iter = tensors_.find(tensor_name);
      if(iter != tensors_.end()){
       auto tensor1 = iter->second;
       const auto & process_group = getTensorProcessGroup(tensor0->getName(),tensor1->getName());
       std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::ADD);
       op->setTensorOperand(tensor0,complex_conj0);
       op->setTensorOperand(tensor1,complex_conj1);
       op->setIndexPattern(addition);
       op->setScalar(0,std::complex<double>(alpha));
       parsed = submit(op,getTensorMapper(process_group));
       if(parsed) parsed = sync(*op);
      }else{
       parsed = true;
       //std::cout << "#ERROR(exatn::NumServer::addTensors): Tensor " << tensor_name << " not found in tensor addition: "
       //          << addition << std::endl;
      }
     }else{
      std::cout << "#ERROR(exatn::NumServer::addTensors): Invalid argument#1 in tensor addition: "
                << addition << std::endl;
     }
    }else{
     parsed = true;
     //std::cout << "#ERROR(exatn::NumServer::addTensors): Tensor " << tensor_name << " not found in tensor addition: "
     //          << addition << std::endl;
    }
   }else{
    std::cout << "#ERROR(exatn::NumServer::addTensors): Invalid argument#0 in tensor addition: "
              << addition << std::endl;
   }
  }else{
   parsed = false;
   std::cout << "#ERROR(exatn::NumServer::addTensors): Invalid number of arguments in tensor addition: "
             << addition << std::endl;
  }
 }else{
  std::cout << "#ERROR(exatn::NumServer::addTensors): Invalid tensor addition: " << addition << std::endl;
 }
 return parsed;
}

template<typename NumericType>
bool NumServer::contractTensors(const std::string & contraction,
                                NumericType alpha)
{
 std::vector<std::string> tensors;
 auto parsed = parse_tensor_network(contraction,tensors);
 if(parsed){
  if(tensors.size() == 3){
   std::string tensor_name;
   std::vector<IndexLabel> indices;
   bool complex_conj0,complex_conj1,complex_conj2;
   parsed = parse_tensor(tensors[0],tensor_name,indices,complex_conj0);
   if(parsed){
    assert(!complex_conj0);
    auto iter = tensors_.find(tensor_name);
    if(iter != tensors_.end()){
     auto tensor0 = iter->second;
     parsed = parse_tensor(tensors[1],tensor_name,indices,complex_conj1);
     if(parsed){
      iter = tensors_.find(tensor_name);
      if(iter != tensors_.end()){
       auto tensor1 = iter->second;
       parsed = parse_tensor(tensors[2],tensor_name,indices,complex_conj2);
       if(parsed){
        iter = tensors_.find(tensor_name);
        if(iter != tensors_.end()){
         auto tensor2 = iter->second;
         const auto & process_group = getTensorProcessGroup(tensor0->getName(),tensor1->getName(),tensor2->getName());
         std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::CONTRACT);
         op->setTensorOperand(tensor0,complex_conj0);
         op->setTensorOperand(tensor1,complex_conj1);
         op->setTensorOperand(tensor2,complex_conj2);
         op->setIndexPattern(contraction);
         op->setScalar(0,std::complex<double>(alpha));
         parsed = submit(op,getTensorMapper(process_group));
        }else{
         parsed = true;
         //std::cout << "#ERROR(exatn::NumServer::contractTensors): Tensor " << tensor_name << " not found in tensor contraction: "
         //          << contraction << std::endl;
        }
       }else{
        std::cout << "#ERROR(exatn::NumServer::contractTensors): Invalid argument#2 in tensor contraction: "
                  << contraction << std::endl;
       }
      }else{
       parsed = true;
       //std::cout << "#ERROR(exatn::NumServer::contractTensors): Tensor " << tensor_name << " not found in tensor contraction: "
       //          << contraction << std::endl;
      }
     }else{
      std::cout << "#ERROR(exatn::NumServer::contractTensors): Invalid argument#1 in tensor contraction: "
                << contraction << std::endl;
     }
    }else{
     parsed = true;
     //std::cout << "#ERROR(exatn::NumServer::contractTensors): Tensor " << tensor_name << " not found in tensor contraction: "
     //          << contraction << std::endl;
    }
   }else{
    std::cout << "#ERROR(exatn::NumServer::contractTensors): Invalid argument#0 in tensor contraction: "
              << contraction << std::endl;
   }
  }else{
   parsed = false;
   std::cout << "#ERROR(exatn::NumServer::contractTensors): Invalid number of arguments in tensor contraction: "
             << contraction << std::endl;
  }
 }else{
  std::cout << "#ERROR(exatn::NumServer::contractTensors): Invalid tensor contraction: " << contraction << std::endl;
 }
 return parsed;
}

template<typename NumericType>
bool NumServer::contractTensorsSync(const std::string & contraction,
                                    NumericType alpha)
{
 std::vector<std::string> tensors;
 auto parsed = parse_tensor_network(contraction,tensors);
 if(parsed){
  if(tensors.size() == 3){
   std::string tensor_name;
   std::vector<IndexLabel> indices;
   bool complex_conj0,complex_conj1,complex_conj2;
   parsed = parse_tensor(tensors[0],tensor_name,indices,complex_conj0);
   if(parsed){
    assert(!complex_conj0);
    auto iter = tensors_.find(tensor_name);
    if(iter != tensors_.end()){
     auto tensor0 = iter->second;
     parsed = parse_tensor(tensors[1],tensor_name,indices,complex_conj1);
     if(parsed){
      iter = tensors_.find(tensor_name);
      if(iter != tensors_.end()){
       auto tensor1 = iter->second;
       parsed = parse_tensor(tensors[2],tensor_name,indices,complex_conj2);
       if(parsed){
        iter = tensors_.find(tensor_name);
        if(iter != tensors_.end()){
         auto tensor2 = iter->second;
         const auto & process_group = getTensorProcessGroup(tensor0->getName(),tensor1->getName(),tensor2->getName());
         std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::CONTRACT);
         op->setTensorOperand(tensor0,complex_conj0);
         op->setTensorOperand(tensor1,complex_conj1);
         op->setTensorOperand(tensor2,complex_conj2);
         op->setIndexPattern(contraction);
         op->setScalar(0,std::complex<double>(alpha));
         parsed = submit(op,getTensorMapper(process_group));
         if(parsed) parsed = sync(*op);
        }else{
         parsed = true;
         //std::cout << "#ERROR(exatn::NumServer::contractTensors): Tensor " << tensor_name << " not found in tensor contraction: "
         //          << contraction << std::endl;
        }
       }else{
        std::cout << "#ERROR(exatn::NumServer::contractTensors): Invalid argument#2 in tensor contraction: "
                  << contraction << std::endl;
       }
      }else{
       parsed = true;
       //std::cout << "#ERROR(exatn::NumServer::contractTensors): Tensor " << tensor_name << " not found in tensor contraction: "
       //          << contraction << std::endl;
      }
     }else{
      std::cout << "#ERROR(exatn::NumServer::contractTensors): Invalid argument#1 in tensor contraction: "
                << contraction << std::endl;
     }
    }else{
     parsed = true;
     //std::cout << "#ERROR(exatn::NumServer::contractTensors): Tensor " << tensor_name << " not found in tensor contraction: "
     //          << contraction << std::endl;
    }
   }else{
    std::cout << "#ERROR(exatn::NumServer::contractTensors): Invalid argument#0 in tensor contraction: "
              << contraction << std::endl;
   }
  }else{
   parsed = false;
   std::cout << "#ERROR(exatn::NumServer::contractTensors): Invalid number of arguments in tensor contraction: "
             << contraction << std::endl;
  }
 }else{
  std::cout << "#ERROR(exatn::NumServer::contractTensors): Invalid tensor contraction: " << contraction << std::endl;
 }
 return parsed;
}

} //namespace exatn

#endif //EXATN_NUM_SERVER_HPP_
