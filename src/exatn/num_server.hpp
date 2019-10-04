/** ExaTN::Numerics: Numerical server
REVISION: 2019/10/04

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Numerical server provides basic tensor network processing functionality:
     + Opening/closing TAProL scopes (top scope 0 "GLOBAL" is open automatically);
     + Creation/destruction of named vector spaces and their named subspaces;
     + Registration/retrieval of external data (class BytePacket);
     + Registration/retrieval of external tensor methods (class TensorFunctor);
     + Submission for processing of individual tensor operations or tensor networks.
     + Higher-level methods for tensor creation, destruction, and operations on them.
 (b) Processing of individual tensor operations or tensor networks has asynchronous semantics:
     Submit TensorOperation/TensorNetwork for processing, then synchronize on the tensor-result.
     Processing of a tensor operation means evaluating the output tensor operand (#0).
     Processing of a tensor network means evaluating the output tensor (#0).
     Synchronization of processing of a tensor operation or tensor network
     means ensuring that the tensor-result (output tensor) has been fully computed.
**/

#ifndef EXATN_NUM_SERVER_HPP_
#define EXATN_NUM_SERVER_HPP_

#include "tensor_basic.hpp"
#include "space_register.hpp"
#include "tensor.hpp"
#include "tensor_op_factory.hpp"
#include "tensor_network.hpp"
#include "tensor_symbol.hpp"

#include "tensor_runtime.hpp"

#include "Identifiable.hpp"
#include "tensor_method.hpp"
#include "functor_init_val.hpp"

#include <memory>
#include <vector>
#include <string>
#include <stack>
#include <map>
#include <future>

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

using TensorMethod = talsh::TensorFunctor<Identifiable>;


class NumServer final {

public:

 NumServer();
 NumServer(const NumServer &) = delete;
 NumServer & operator=(const NumServer &) = delete;
 NumServer(NumServer &&) noexcept = delete;
 NumServer & operator=(NumServer &&) noexcept = delete;
 ~NumServer();

 /** Reconfigures tensor runtime implementation. **/
 void reconfigureTensorRuntime(const std::string & dag_executor_name,
                               const std::string & node_executor_name);

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
 void submit(std::shared_ptr<TensorOperation> operation);
 /** Submits a tensor network for processing (evaluating the tensor-result). **/
 void submit(TensorNetwork & network);
 void submit(std::shared_ptr<TensorNetwork> network);

 /** Synchronizes all update operations on a given tensor. **/
 bool sync(const Tensor & tensor,
           bool wait = true);
 /** Synchronizes execution of a specific tensor operation. **/
 bool sync(TensorOperation & operation,
           bool wait = true);
 /** Synchronizes execution of a specific tensor network. **/
 bool sync(TensorNetwork & network,
           bool wait = true);

 /** HIGHER-LEVEL WRAPPERS **/

 /** Synchronizes all outstanding update operations on a given tensor. **/
 bool sync(const std::string & name, //in: tensor name
           bool wait = true);        //in: wait versus test for completion

 /** Returns the reference to the actual tensor object. **/
 Tensor & getTensorRef(const std::string & name); //in: tensor name

 /** Returns the tensor element type. **/
 TensorElementType getTensorElementType(const std::string & name) const; //in: tensor name

 /** Declares, registers and actually creates a tensor via processing backend.
     See numerics::Tensor constructors for different creation options. **/
 template <typename... Args>
 bool createTensor(const std::string & name,       //in: tensor name
                   TensorElementType element_type, //in: tensor element type
                   Args&&... args);                //in: other arguments for Tensor ctor

 /** Destroys a tensor, including its backend representation. **/
 bool destroyTensor(const std::string & name); //in: tensor name

 /** Initializes a tensor to some scalar value. **/
 template<typename NumericType>
 bool initTensor(const std::string & name, //in: tensor name
                 NumericType value,        //in: scalar value
                 bool async = true);       //in: asynchronisity

 /** Transforms (updates) a tensor according to a user-defined tensor functor. **/
 bool transformTensor(const std::string & name,              //in: tensor name
                      std::shared_ptr<TensorMethod> functor, //in: functor defining tensor transformation
                      bool async = true);                    //in: asynchronisity

 /** Performs tensor addition: tensor0 += tensor1 * alpha **/
 template<typename NumericType>
 bool addTensors(const std::string & addition, //in: symbolic tensor addition specification
                 NumericType alpha);           //in: alpha prefactor

 /** Performs tensor contraction: tensor0 += tensor1 * tensor2 * alpha **/
 template<typename NumericType>
 bool contractTensors(const std::string & contraction, //in: symbolic tensor contraction specification
                      NumericType alpha);              //in: alpha prefactor

 /** Performs a full evaluation of a tensor network. **/
 bool evaluateTensorNetwork(const std::string & name,     //in: tensor network name
                            const std::string & network); //in: symbolic tensor network specification

 /** Returns a locally stored tensor slice (talsh::Tensor) providing access to tensor elements.
     This slice will be extracted from the exatn::numerics::Tensor implementation as a copy.
     The returned future becomes ready once the execution thread has retrieved the slice copy. **/
 std::future<std::shared_ptr<talsh::Tensor>> getLocalTensor(std::shared_ptr<Tensor> tensor, //in: exatn::numerics::Tensor to get slice of (by copy)
                           const std::vector<std::pair<DimOffset,DimExtent>> & slice_spec); //in: tensor slice specification

private:

 numerics::SpaceRegister space_register_; //register of vector spaces and their named subspaces
 std::unordered_map<std::string,SpaceId> subname2id_; //maps a subspace name to its parental vector space id

 std::unordered_map<std::string,std::shared_ptr<Tensor>> tensors_; //registered tensors

 std::map<std::string,std::shared_ptr<TensorMethod>> ext_methods_; //external tensor methods
 std::map<std::string,std::shared_ptr<BytePacket>> ext_data_; //external data

 std::stack<std::pair<std::string,ScopeId>> scopes_; //TAProL scope stack: {Scope name, Scope Id}

 TensorOpFactory * tensor_op_factory_; //tensor operation factory

 std::shared_ptr<runtime::TensorRuntime> tensor_rt_; //tensor runtime (for actual execution of tensor operations)
};

/** Numerical service singleton (numerical server) **/
extern std::shared_ptr<NumServer> numericalServer;

//TEMPLATE DEFINITIONS:
template <typename... Args>
bool NumServer::createTensor(const std::string & name, TensorElementType element_type, Args&&... args)
{
 auto res = tensors_.emplace(std::make_pair(name,std::shared_ptr<Tensor>(new Tensor(name,args...))));
 if(res.second){
  std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::CREATE);
  op->setTensorOperand((res.first)->second);
  std::dynamic_pointer_cast<numerics::TensorOpCreate>(op)->resetTensorElementType(element_type);
  submit(op);
 }else{
  std::cout << "#ERROR(exatn::NumServer::createTensor): Tensor " << name << " already exists!" << std::endl;
 }
 return res.second;
}

template<typename NumericType>
bool NumServer::initTensor(const std::string & name,
                           NumericType value,
                           bool async)
{
 return transformTensor(name,std::shared_ptr<TensorMethod>(new numerics::FunctorInitVal(value)),async);
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
   bool complex_conj;
   parsed = parse_tensor(tensors[0],tensor_name,indices,complex_conj);
   if(parsed){
    auto iter = tensors_.find(tensor_name);
    if(iter != tensors_.end()){
     auto tensor0 = iter->second;
     parsed = parse_tensor(tensors[1],tensor_name,indices,complex_conj);
     if(parsed){
      iter = tensors_.find(tensor_name);
      if(iter != tensors_.end()){
       auto tensor1 = iter->second;
       std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::ADD);
       op->setTensorOperand(tensor0);
       op->setTensorOperand(tensor1);
       op->setIndexPattern(addition);
       op->setScalar(0,std::complex<double>(alpha));
       submit(op);
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
   bool complex_conj;
   parsed = parse_tensor(tensors[0],tensor_name,indices,complex_conj);
   if(parsed){
    auto iter = tensors_.find(tensor_name);
    if(iter != tensors_.end()){
     auto tensor0 = iter->second;
     parsed = parse_tensor(tensors[1],tensor_name,indices,complex_conj);
     if(parsed){
      iter = tensors_.find(tensor_name);
      if(iter != tensors_.end()){
       auto tensor1 = iter->second;
       parsed = parse_tensor(tensors[2],tensor_name,indices,complex_conj);
       if(parsed){
        iter = tensors_.find(tensor_name);
        if(iter != tensors_.end()){
         auto tensor2 = iter->second;
         std::shared_ptr<TensorOperation> op = tensor_op_factory_->createTensorOp(TensorOpCode::CONTRACT);
         op->setTensorOperand(tensor0);
         op->setTensorOperand(tensor1);
         op->setTensorOperand(tensor2);
         op->setIndexPattern(contraction);
         op->setScalar(0,std::complex<double>(alpha));
         submit(op);
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
