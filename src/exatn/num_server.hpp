/** ExaTN::Numerics: Numerical server
REVISION: 2019/09/03

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

/** Rationale:
 (a) Numerical server provides basic tensor network processing functionality:
     + Opening/closing TAProL scopes (top scope 0 "GLOBAL" is open automatically);
     + Creation/destruction of named vector spaces and their named subspaces;
     + Registration/retrieval of external data (class BytePacket);
     + Registration/retrieval of external tensor methods (class TensorMethod);
     + Submission for processing of individual tensor operations or tensor networks.
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
#include "tensor_operation.hpp"
#include "tensor_network.hpp"

#include "tensor_runtime.hpp"

#include "tensor_method.hpp"
#include "Identifiable.hpp"

#include <memory>
#include <string>
#include <stack>
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
using numerics::TensorNetwork;


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
 void registerTensorMethod(std::shared_ptr<TensorMethod<Identifiable>> method);

 /** Retrieves a registered external tensor method. **/
 std::shared_ptr<TensorMethod<Identifiable>> getTensorMethod(const std::string & tag);

 /** Registers an external data packet. **/
 void registerExternalData(const std::string & tag, std::shared_ptr<BytePacket> packet);

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
 void destroySubspace(SubspaceId subspace_id);

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
           bool wait = false);
 /** Synchronizes execution of a specific tensor operation. **/
 bool sync(TensorOperation & operation,
           bool wait = false);
 /** Synchronizes execution of a specific tensor network. **/
 bool sync(TensorNetwork & network,
           bool wait = false);

private:

 numerics::SpaceRegister space_register_; //register of vector spaces and their named subspaces
 std::unordered_map<std::string,SpaceId> subname2id_; //maps a subspace name to its parental vector space id

 std::map<std::string,std::shared_ptr<TensorMethod<Identifiable>>> ext_methods_; //external tensor methods
 std::map<std::string,std::shared_ptr<BytePacket>> ext_data_; //external data

 std::stack<std::pair<std::string,ScopeId>> scopes_; //TAProL scope stack: {Scope name, Scope Id}

 std::shared_ptr<runtime::TensorRuntime> tensor_rt_; //tensor runtime (for actual execution of tensor operations)
};

/** Numerical service singleton (numerical server) **/
extern std::shared_ptr<NumServer> numericalServer;

} //namespace exatn

#endif //EXATN_NUM_SERVER_HPP_
