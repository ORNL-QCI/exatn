/** ExaTN::Numerics: Numerical server
REVISION: 2020/04/07

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

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
     Processing of a tensor operation means evaluating the output tensor operand (#0).
     Processing of a tensor network means evaluating the output tensor (#0),
     which is automatically initialized to zero before the evaluation.
     Processing of a tensor network expansion means evaluating all constituent
     tensor network components and accumulating them into the accumulator tensor
     with their respective prefactors. Synchronization of processing of a tensor operation,
     tensor network or tensor network expansion means ensuring that the tensor-result,
     either the output tensor or accumulator tensor, has been fully computed.
 (c) Namespace exatn introduces a number of aliases for types imported from exatn::numerics.
     Importantly exatn::TensorMethod, which is talsh::TensorFunctor<Indentifiable>,
     defines the interface which needs to be implemented by the application in order
     to perform an arbitrary custom unary transform operation on exatn::Tensor.
     This is the only portable way to arbitrarily modify tensor content.
**/

#ifndef EXATN_NUM_SERVER_HPP_
#define EXATN_NUM_SERVER_HPP_

#include "tensor_basic.hpp"
#include "space_register.hpp"
#include "tensor.hpp"
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
#include "functor_scale.hpp"

#include <memory>
#include <vector>
#include <string>
#include <stack>
#include <list>
#include <map>

using exatn::Identifiable;

namespace exatn{

using numerics::VectorSpace;
using numerics::Subspace;
using numerics::TensorShape;
using numerics::TensorSignature;
using numerics::TensorLeg;
using numerics::Tensor;
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
using numerics::FunctorScale;

using TensorMethod = talsh::TensorFunctor<Identifiable>;


class NumServer final {

public:

#ifdef MPI_ENABLED
 NumServer(const MPICommProxy & communicator,                               //MPI communicator proxy
           const std::string & graph_executor_name = "eager-dag-executor",  //DAG executor kind
           const std::string & node_executor_name = "talsh-node-executor"); //DAG node executor kind
#else
 NumServer(const std::string & graph_executor_name = "eager-dag-executor",  //DAG executor kind
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
                               const std::string & dag_executor_name,
                               const std::string & node_executor_name);
#else
 void reconfigureTensorRuntime(const std::string & dag_executor_name,
                               const std::string & node_executor_name);
#endif

 /** Resets the tensor contraction sequence optimizer that is
     invoked when evaluating tensor networks. **/
 void resetContrSeqOptimizer(const std::string & optimizer_name);

 /** Resets the runtime logging level (0:none). **/
 void resetRuntimeLoggingLevel(int level = 0);

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


 /** Submits an individual tensor operation for processing. **/
 bool submit(std::shared_ptr<TensorOperation> operation); //in: tensor operation for numerical evaluation

 /** Submits a tensor network for processing (evaluating the output tensor-result).
     If the output (result) tensor has not been created yet, it will be created and
     initialized to zero automatically, and later destroyed automatically when no longer needed.
     By default all parallel processes will be processing the tensor network,
     otherwise the desired process subset needs to be explicitly specified. **/
 bool submit(TensorNetwork & network);                       //in: tensor network for numerical evaluation
 bool submit(std::shared_ptr<TensorNetwork> network);        //in: tensor network for numerical evaluation
 bool submit(TensorNetwork & network,                        //in: tensor network for numerical evaluation
             const std::vector<unsigned int> & process_set); //in: chosen set of parallel processes
 bool submit(std::shared_ptr<TensorNetwork> network,         //in: tensor network for numerical evaluation
             const std::vector<unsigned int> & process_set); //in: chosen set of parallel processes

 /** Submits a tensor network expansion for processing (evaluating output tensors of all
     constituting tensor networks and accumualting them in the provided accumulator tensor).
     Synchronization of the tensor expansion evaluation is done via syncing on the accumulator
     tensor. By default all parallel processes will be processing the tensor network,
     otherwise the desired process subset needs to be explicitly specified. **/
 bool submit(TensorExpansion & expansion,                 //in: tensor expansion for numerical evaluation
             std::shared_ptr<Tensor> accumulator);        //inout: tensor accumulator (result)
 bool submit(std::shared_ptr<TensorExpansion> expansion,  //in: tensor expansion for numerical evaluation
             std::shared_ptr<Tensor> accumulator);        //inout: tensor accumulator (result)
 bool submit(TensorExpansion & expansion,                    //in: tensor expansion for numerical evaluation
             std::shared_ptr<Tensor> accumulator,            //inout: tensor accumulator (result)
             const std::vector<unsigned int> & process_set); //in: chosen set of parallel processes
 bool submit(std::shared_ptr<TensorExpansion> expansion,     //in: tensor expansion for numerical evaluation
             std::shared_ptr<Tensor> accumulator,            //inout: tensor accumulator (result)
             const std::vector<unsigned int> & process_set); //in: chosen set of parallel processes

 /** Synchronizes all update operations on a given tensor. **/
 bool sync(const Tensor & tensor,
           bool wait = true);
 /** Synchronizes execution of a specific tensor operation. **/
 bool sync(TensorOperation & operation,
           bool wait = true);
 /** Synchronizes execution of a specific tensor network. **/
 bool sync(TensorNetwork & network,
           bool wait = true);
 /** Synchronizes execution of all outstanding tensor operations. **/
 bool sync(bool wait = true);

 /** HIGHER-LEVEL WRAPPERS **/

 /** Synchronizes all outstanding update operations on a given tensor. **/
 bool sync(const std::string & name, //in: tensor name
           bool wait = true);        //in: wait versus test for completion

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

 /** Declares, registers and actually creates a tensor via processing backend.
     See numerics::Tensor constructors for different creation options. **/
 template <typename... Args>
 bool createTensor(const std::string & name,       //in: tensor name
                   TensorElementType element_type, //in: tensor element type
                   Args&&... args);                //in: other arguments for Tensor ctor

 bool createTensor(std::shared_ptr<Tensor> tensor,  //in: existing declared tensor
                   TensorElementType element_type); //in: tensor element type

 template <typename... Args>
 bool createTensorSync(const std::string & name,       //in: tensor name
                       TensorElementType element_type, //in: tensor element type
                       Args&&... args);                //in: other arguments for Tensor ctor

 bool createTensorSync(std::shared_ptr<Tensor> tensor,  //in: existing declared tensor
                       TensorElementType element_type); //in: tensor element type

 /** Destroys a tensor, including its backend representation. **/
 bool destroyTensor(const std::string & name); //in: tensor name

 bool destroyTensorSync(const std::string & name); //in: tensor name

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

 /** Initializes a tensor to some random value. **/
 bool initTensorRnd(const std::string & name); //in: tensor name

 bool initTensorRndSync(const std::string & name); //in: tensor name

 /** Scales a tensor by a scalar value. **/
 template<typename NumericType>
 bool scaleTensor(const std::string & name, //in: tensor name
                  NumericType value);       //in: scalar value

 template<typename NumericType>
 bool scaleTensorSync(const std::string & name, //in: tensor name
                      NumericType value);       //in: scalar value

 /** Transforms (updates) a tensor according to a user-defined tensor functor. **/
 bool transformTensor(const std::string & name,               //in: tensor name
                      std::shared_ptr<TensorMethod> functor); //in: functor defining tensor transformation

 bool transformTensorSync(const std::string & name,               //in: tensor name
                          std::shared_ptr<TensorMethod> functor); //in: functor defining tensor transformation

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
      S(i,j) is the middle SVD factor (the diagonal with singular values). **/
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
     the middle factor with singular values. The symbolic tensor contraction
     specification specifies the decomposition. **/
 bool orthogonalizeTensorSVD(const std::string & contraction); //in: two-factor symbolic tensor contraction specification

 bool orthogonalizeTensorSVDSync(const std::string & contraction); //in: two-factor symbolic tensor contraction specification

 /** Orthogonalizes a tensor over its isometric dimensions via the modified Gram-Schmidt procedure.  **/
 bool orthogonalizeTensorMGS(const std::string & name); //in: tensor name

 bool orthogonalizeTensorMGSSync(const std::string & name); //in: tensor name

 /** Performs a full evaluation of a tensor network based on the symbolic
     specification involving already created tensors (including the output). **/
 bool evaluateTensorNetwork(const std::string & name,     //in: tensor network name
                            const std::string & network); //in: symbolic tensor network specification

 bool evaluateTensorNetworkSync(const std::string & name,     //in: tensor network name
                                const std::string & network); //in: symbolic tensor network specification

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

private:

 void destroyOrphanedTensors();

 std::shared_ptr<numerics::SpaceRegister> space_register_; //register of vector spaces and their named subspaces
 std::unordered_map<std::string,SpaceId> subname2id_; //maps a subspace name to its parental vector space id

 std::unordered_map<std::string,std::shared_ptr<Tensor>> tensors_; //registered tensors (by CREATE operation)
 std::list<std::shared_ptr<Tensor>> implicit_tensors_; //tensors created implicitly by the runtime (for garbage collection)

 std::string contr_seq_optimizer_; //tensor contraction sequence optimizer invoked when evaluating tensor networks

 std::map<std::string,std::shared_ptr<TensorMethod>> ext_methods_; //external tensor methods
 std::map<std::string,std::shared_ptr<BytePacket>> ext_data_; //external data

 std::stack<std::pair<std::string,ScopeId>> scopes_; //TAProL scope stack: {Scope name, Scope Id}

 TensorOpFactory * tensor_op_factory_; //tensor operation factory (non-owning pointer)

 int num_processes_; //total number of parallel processes
 int process_rank_; //rank of the current parallel process
 std::shared_ptr<runtime::TensorRuntime> tensor_rt_; //tensor runtime (for actual execution of tensor operations)
};

/** Numerical service singleton (numerical server) **/
extern std::shared_ptr<NumServer> numericalServer;

//TEMPLATE DEFINITIONS:
template <typename... Args>
bool NumServer::createTensor(const std::string & name,
                             TensorElementType element_type,
                             Args&&... args)
{
 std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::CREATE);
 op->setTensorOperand(std::make_shared<Tensor>(name,std::forward<Args>(args)...));
 std::dynamic_pointer_cast<numerics::TensorOpCreate>(op)->resetTensorElementType(element_type);
 auto submitted = submit(op);
 return submitted;
}

template <typename... Args>
bool NumServer::createTensorSync(const std::string & name,
                                 TensorElementType element_type,
                                 Args&&... args)
{
 std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::CREATE);
 op->setTensorOperand(std::make_shared<Tensor>(name,std::forward<Args>(args)...));
 std::dynamic_pointer_cast<numerics::TensorOpCreate>(op)->resetTensorElementType(element_type);
 auto submitted = submit(op);
 if(submitted) submitted = sync(*op);
 return submitted;
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
       std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::ADD);
       op->setTensorOperand(tensor0,complex_conj0);
       op->setTensorOperand(tensor1,complex_conj1);
       op->setIndexPattern(addition);
       op->setScalar(0,std::complex<double>(alpha));
       parsed = submit(op);
      }else{
       parsed = false;
       std::cout << "#ERROR(exatn::NumServer::addTensors): Tensor " << tensor_name << " not found in tensor addition: "
                 << addition << std::endl;
      }
     }else{
      std::cout << "#ERROR(exatn::NumServer::addTensors): Invalid argument#1 in tensor addition: "
                << addition << std::endl;
     }
    }else{
     parsed = false;
     std::cout << "#ERROR(exatn::NumServer::addTensors): Tensor " << tensor_name << " not found in tensor addition: "
               << addition << std::endl;
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
       std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::ADD);
       op->setTensorOperand(tensor0,complex_conj0);
       op->setTensorOperand(tensor1,complex_conj1);
       op->setIndexPattern(addition);
       op->setScalar(0,std::complex<double>(alpha));
       parsed = submit(op);
       if(parsed) parsed = sync(*op);
      }else{
       parsed = false;
       std::cout << "#ERROR(exatn::NumServer::addTensors): Tensor " << tensor_name << " not found in tensor addition: "
                 << addition << std::endl;
      }
     }else{
      std::cout << "#ERROR(exatn::NumServer::addTensors): Invalid argument#1 in tensor addition: "
                << addition << std::endl;
     }
    }else{
     parsed = false;
     std::cout << "#ERROR(exatn::NumServer::addTensors): Tensor " << tensor_name << " not found in tensor addition: "
               << addition << std::endl;
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
         std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::CONTRACT);
         op->setTensorOperand(tensor0,complex_conj0);
         op->setTensorOperand(tensor1,complex_conj1);
         op->setTensorOperand(tensor2,complex_conj2);
         op->setIndexPattern(contraction);
         op->setScalar(0,std::complex<double>(alpha));
         parsed = submit(op);
        }else{
         parsed = false;
         std::cout << "#ERROR(exatn::NumServer::contractTensors): Tensor " << tensor_name << " not found in tensor contraction: "
                   << contraction << std::endl;
        }
       }else{
        std::cout << "#ERROR(exatn::NumServer::contractTensors): Invalid argument#2 in tensor contraction: "
                  << contraction << std::endl;
       }
      }else{
       parsed = false;
       std::cout << "#ERROR(exatn::NumServer::contractTensors): Tensor " << tensor_name << " not found in tensor contraction: "
                 << contraction << std::endl;
      }
     }else{
      std::cout << "#ERROR(exatn::NumServer::contractTensors): Invalid argument#1 in tensor contraction: "
                << contraction << std::endl;
     }
    }else{
     parsed = false;
     std::cout << "#ERROR(exatn::NumServer::contractTensors): Tensor " << tensor_name << " not found in tensor contraction: "
               << contraction << std::endl;
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
         std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::CONTRACT);
         op->setTensorOperand(tensor0,complex_conj0);
         op->setTensorOperand(tensor1,complex_conj1);
         op->setTensorOperand(tensor2,complex_conj2);
         op->setIndexPattern(contraction);
         op->setScalar(0,std::complex<double>(alpha));
         parsed = submit(op);
         if(parsed) parsed = sync(*op);
        }else{
         parsed = false;
         std::cout << "#ERROR(exatn::NumServer::contractTensors): Tensor " << tensor_name << " not found in tensor contraction: "
                   << contraction << std::endl;
        }
       }else{
        std::cout << "#ERROR(exatn::NumServer::contractTensors): Invalid argument#2 in tensor contraction: "
                  << contraction << std::endl;
       }
      }else{
       parsed = false;
       std::cout << "#ERROR(exatn::NumServer::contractTensors): Tensor " << tensor_name << " not found in tensor contraction: "
                 << contraction << std::endl;
      }
     }else{
      std::cout << "#ERROR(exatn::NumServer::contractTensors): Invalid argument#1 in tensor contraction: "
                << contraction << std::endl;
     }
    }else{
     parsed = false;
     std::cout << "#ERROR(exatn::NumServer::contractTensors): Tensor " << tensor_name << " not found in tensor contraction: "
               << contraction << std::endl;
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
